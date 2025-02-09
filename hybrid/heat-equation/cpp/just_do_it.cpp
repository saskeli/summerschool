#include <mpi.h>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "pngwriter.h"

#ifdef DEBUG
static const constexpr bool debug = true;
#else
static const constexpr bool debug = false;
#endif

static const constexpr double a = 0.5;
static const constexpr double dx = 0.01;
static const constexpr double dy = 0.01;
static const constexpr double dt =
    dx * dx * dy * dy / (2.0 * a * (dx * dx + dy * dy));
static const constexpr double inv_dx2 = 1.0 / (dx * dx);
static const constexpr double inv_dy2 = 1.0 / (dy * dy);
static int dims[2];
static const int periods[] = {0, 0};
static MPI_Comm comm;
static int rank, size;
static double* raw_subA;
static double* raw_subB;
static double** data_subA;
static double** data_subB;
static uint32_t nx = 20;
static uint32_t ny = 20;
static uint32_t iters = 20;
static uint32_t stepping = 1;
static uint32_t snx, sny;
MPI_Datatype raw_block, sub_block, row, col;
static int* raw_offsets;
static int* send_counts;
static const int counts[] = {1, 1, 1, 1};

void init_edges() {
    for (uint32_t i = 0; i < snx + 2; i++) {
        data_subA[0][i] = 100.0;
        data_subA[sny + 1][i] = 100.0;
    }
    for (uint32_t i = 1; i <= sny; i++) {
        data_subA[i][0] = 100.0;
        data_subA[i][snx + 1] = 100.0;
    }
}

void meta_and_sub(double* senduf) {
    uint32_t meta[] = {nx, ny, iters, stepping};
    MPI_Bcast(meta, 4, MPI_UINT32_T, 0, comm);
    nx = meta[0];
    ny = meta[1];
    iters = meta[2];
    stepping = meta[3];
    if (nx % dims[0] || ny % dims[1]) {
        if (rank == 0) {
            std::cerr << "(" << nx << ", " << ny << ") is not divisible by ("
                      << dims[0] << ", " << dims[1] << ")" << std::endl;
        }
        MPI_Abort(comm, 1);
        exit(1);
    }
    snx = nx / dims[1];
    sny = ny / dims[0];
    if (rank == 0) {
        int ln, rn;
        MPI_Cart_shift(comm, 1, 1, &ln, &rn);
        std::cout << "Split to " << size << " tasks\n"
                  << "With " << sny << " x " << snx << " sub-block size\n"
                  << "In " << dims[0] << " x " << dims[1] << " grid\n"
                  << "To my left is " << ln << " and right is " << rn
                  << std::endl;
    }
    raw_subA = (double*)malloc((2 + snx) * (2 + sny) * sizeof(double));
    data_subA = (double**)malloc((2 + sny) * sizeof(double*));
    raw_subB = (double*)malloc((2 + snx) * (2 + sny) * sizeof(double));
    data_subB = (double**)malloc((2 + sny) * sizeof(double*));
    for (uint32_t r = 0; r < sny + 2; r++) {
        data_subA[r] = raw_subA + (r * (snx + 2));
        data_subB[r] = raw_subB + (r * (snx + 2));
    }
    MPI_Datatype tmp;
    MPI_Type_vector(sny, snx, nx, MPI_DOUBLE, &tmp);
    MPI_Type_create_resized(tmp, 0, sizeof(double), &raw_block);
    MPI_Type_commit(&raw_block);
    MPI_Type_vector(sny, snx, snx + 2, MPI_DOUBLE, &sub_block);
    MPI_Type_commit(&sub_block);
    MPI_Type_contiguous(snx, MPI_DOUBLE, &row);
    MPI_Type_commit(&row);
    MPI_Type_vector(sny, 1, snx + 2, MPI_DOUBLE, &col);
    MPI_Type_commit(&col);

    raw_offsets = (int*)malloc(size * sizeof(int));
    send_counts = (int*)malloc(size * sizeof(int));
    for (int r = 0; r < dims[0]; r++) {
        for (int c = 0; c < dims[1]; c++) {
            raw_offsets[r * dims[1] + c] = r * sny * nx + c * snx;
            send_counts[r * dims[1] + c] = 1;
        }
    }

    if (rank == 0) {
        std::cout << "Raw offsets:\n";
        for (int r = 0; r < dims[0]; r++) {
            for (int c = 0; c < dims[1]; c++) {
                std::cout << r * dims[1] + c << ": "
                          << raw_offsets[r * dims[1] + c] << " ";
            }
            std::cout << std::endl;
        }
    }

    MPI_Scatterv(senduf, send_counts, raw_offsets, raw_block,
                 raw_subA + snx + 3, 1, sub_block, 0, comm);
    init_edges();
}

void write(double* data, uint32_t t) {
    std::ostringstream filename_stream;
    filename_stream << "heat_" << std::setw(4) << std::setfill('0') << t
                    << ".png";
    std::string filename = filename_stream.str();
    save_png(data, ny, nx, filename.c_str(), 'c');
}

double* read_file(char const* fname) {
    std::ifstream in(fname);
    char c;
    in >> c >> nx >> ny;
    double* raw = (double*)malloc(nx * ny * sizeof(double));
    for (uint32_t i = 0; i < nx * ny; i++) {
        in >> raw[i];
    }
    return raw;
}

double* defdef() {
    uint32_t cut = nx / 10;
    double* raw = (double*)calloc(nx * ny, sizeof(double));
    std::fill(raw, raw + nx * cut, double(95));
    for (uint32_t i = cut; i < ny - cut; i++) {
        double* rp = raw + nx * i;
        std::fill(rp, rp + (nx / 2 - cut), double(95));
        std::fill(rp + nx / 2 + cut, rp + nx, double(95));
    }
    std::fill(raw + nx * (ny - cut), raw + nx * ny, double(95));
    return raw;
}

void simulate(double* raw) {
    MPI_Aint sdisps[] = {snx + 3, sny * (snx + 2) + 1, snx + 3, 2 * snx + 2};
    MPI_Aint rdisps[] = {1, (sny + 1) * (snx + 2) + 1, snx + 2, 2 * snx + 3};
    for (uint32_t i = 0; i < 4; i++) {
        sdisps[i] = sizeof(double) * sdisps[i];
        rdisps[i] = sizeof(double) * rdisps[i];
    }
    MPI_Datatype types[] = {row, row, col, col};
    uint32_t nc = 0;
#pragma omp parallel shared(raw_subA, raw_subB, data_subA, data_subB, sdisps, \
                                types, rdisps, counts, comm, raw)
    {
        for (uint32_t t = 0; t < iters; t++) {
            MPI_Request req;
#pragma omp master
            {
                MPI_Ineighbor_alltoallw(raw_subA, counts, sdisps, types,
                                        raw_subA, counts, rdisps, types, comm,
                                        &req);
            }

#pragma omp for
            for (uint32_t r = 2; r < sny; r++) {
                for (uint32_t c = 2; c < snx; c++) {
                    data_subB[r][c] =
                        data_subA[r][c] +
                        a * dt *
                            ((data_subA[r + 1][c] - 2.0 * data_subA[r][c] +
                              data_subA[r - 1][c]) *
                                 inv_dx2 +
                             (data_subA[r][c + 1] - 2.0 * data_subA[r][c] +
                              data_subA[r][c - 1]) *
                                 inv_dy2);
                }
            }

#pragma omp master
            {
                MPI_Status stat;
                MPI_Wait(&req, &stat);
            }
#pragma omp barrier

#pragma omp task
            {
                for (uint32_t i = 1; i <= snx; i++) {
                    data_subB[1][i] =
                        data_subA[1][i] +
                        a * dt *
                            ((data_subA[1 + 1][i] - 2.0 * data_subA[1][i] +
                              data_subA[1 - 1][i]) *
                                 inv_dx2 +
                             (data_subA[1][i + 1] - 2.0 * data_subA[1][i] +
                              data_subA[1][i - 1]) *
                                 inv_dy2);
                }
            }
#pragma omp task
            {
                for (uint32_t i = 1; i <= snx; i++) {
                    data_subB[sny][i] =
                        data_subA[sny][i] +
                        a * dt *
                            ((data_subA[sny + 1][i] - 2.0 * data_subA[sny][i] +
                              data_subA[sny - 1][i]) *
                                 inv_dx2 +
                             (data_subA[sny][i + 1] - 2.0 * data_subA[sny][i] +
                              data_subA[sny][i - 1]) *
                                 inv_dy2);
                }
            }
#pragma omp task
            {
                for (uint32_t i = 1; i <= sny; i++) {
                    data_subB[i][1] =
                        data_subA[i][1] +
                        a * dt *
                            ((data_subA[i + 1][1] - 2.0 * data_subA[i][1] +
                              data_subA[i - 1][1]) *
                                 inv_dx2 +
                             (data_subA[i][1 + 1] - 2.0 * data_subA[i][1] +
                              data_subA[i][1 - 1]) *
                                 inv_dy2);
                }
            }
#pragma omp task
            {
                for (uint32_t i = 1; i <= sny; i++) {
                    data_subB[i][snx] =
                        data_subA[i][snx] +
                        a * dt *
                            ((data_subA[i + 1][snx] - 2.0 * data_subA[i][snx] +
                              data_subA[i - 1][snx]) *
                                 inv_dx2 +
                             (data_subA[i][snx + 1] - 2.0 * data_subA[i][snx] +
                              data_subA[i][snx - 1]) *
                                 inv_dy2);
                }
            }
#pragma omp barrier
            if (nc == t) {
#pragma omp master
                {
                    MPI_Gatherv(raw_subB + snx + 3, 1, sub_block, raw, counts,
                                raw_offsets, raw_block, 0, comm);
                    if (rank == 0) {
                        write(raw, t);
                    }
                    MPI_Barrier(comm);
                }
                nc += stepping;
            }
        }
        std::swap(raw_subA, raw_subB);
        std::swap(data_subA, data_subB);
#pragma omp barrier
    }
    free(data_subA);
    free(data_subB);
    free(raw_subA);
    free(raw_subB);
    free(raw_offsets);
    free(send_counts);
}

void slave() {
    meta_and_sub(nullptr);
    simulate(nullptr);
    MPI_Finalize();
    exit(0);
}

int main(int argc, char* argv[]) {
    int threading_support;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &threading_support);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Dims_create(size, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm);
    MPI_Comm_rank(comm, &rank);
    if (rank != 0) {
        slave();
    }
    double* raw;
    if (argc > 1) {
        raw = read_file(argv[1]);
    } else {
        raw = defdef();
    }
    if (argc > 2) {
        iters = atoll(argv[2]);
    }
    if (argc > 3) {
        stepping = atoll(argv[3]);
    }
    std::cout << "Debug " << (debug ? "enabled" : "disabled") << std::endl;
    std::cout << "Got " << ny << " x " << nx << " input array\n"
              << "Iterations: " << iters << "\n"
              << "Stepping: " << stepping << std::endl;
#ifdef _OPENMP
    std::cout << "I have " << omp_get_max_threads() << " threads" << std::endl;
#endif

    meta_and_sub(raw);

    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::nanoseconds;

    auto start = high_resolution_clock::now();
    simulate(raw);
    auto end = high_resolution_clock::now();
    double time = duration_cast<nanoseconds>(end - start).count();
    std::cout << "Simulation took " << time / 1000000 << "ms" << std::endl;
    free(raw);
    MPI_Finalize();
    return 0;
}
