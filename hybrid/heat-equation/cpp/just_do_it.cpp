#include <mpi.h>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "pngwriter.h"

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
static double* raw_sub;
static double** data_sub;
static uint32_t nx = 200;
static uint32_t ny = 200;
static uint32_t iters = 600;
static uint32_t stepping = 100;
static uint32_t snx, sny;
MPI_Datatype raw_block, sub_block, row, col;
static int* raw_offsets;
static int* send_counts;
static const int counts[] = {1, 1, 1, 1};

void init_edges() {
#pragma omp parallel
    {
#pragma omp task
        {
            for (uint32_t i = 0; i < nx + 2; i++) {
                data_sub[0][i] = 95.0;
            }
        }
#pragma omp task
        {
            for (uint32_t i = 0; i < nx + 2; i++) {
                data_sub[ny + 1][i] = 95.0;
            }
        }
#pragma omp task
        {
            for (uint32_t i = 1; i <= ny; i++) {
                data_sub[i][0] = 95.0;
            }
        }
#pragma omp task
        {
            for (uint32_t i = 1; i <= ny; i++) {
                data_sub[i][nx + 1] = 95.0;
            }
        }
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
    snx = nx / dims[0];
    sny = ny / dims[1];
    if (rank == 0) {
        std::cout << "Split to " << size << " tasks\n"
                  << "With " << sny << " x " << snx << " sub-block size\n"
                  << "In " << dims[0] << " x " << dims[1] << " grid" << std::endl;
    }
    raw_sub = (double*)malloc((2 + snx) * (2 + sny) * sizeof(double));
    data_sub = (double**)malloc((2 + sny) * sizeof(double*));
    for (uint32_t r = 0; r < sny + 2; r++) {
        data_sub[r] = raw_sub + (r * (snx + 2));
    }
    MPI_Datatype tmp;
    MPI_Type_vector(sny, snx, nx, MPI_DOUBLE, &tmp);
    MPI_Type_create_resized(tmp, 0, sizeof(MPI_DOUBLE), &raw_block);
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
            raw_offsets[r * dims[1] + c] = r * sny * nx + c * nx;
            send_counts[r * dims[1] + c] = 1;
        }
    }

    MPI_Scatterv(senduf, send_counts, raw_offsets, sub_block, raw_sub, 1,
                 raw_block, 0, comm);
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
    double* raw = (double*)calloc(nx * ny, sizeof(double));
    std::fill(raw, raw + nx * 25, double(95));
    for (uint32_t i = 25; i < ny - 25; i++) {
        double* rp = raw + nx * i;
        std::fill(rp, rp + 75, double(95));
        std::fill(rp + 125, rp + nx, double(95));
    }
    std::fill(raw + nx * (ny - 25), raw + nx * ny, double(95));
    return raw;
}

void simulate(double* raw) {
    MPI_Aint sdisps[] = {sny * (snx + 2) + 1, snx + 3, snx + 3, 2 * snx + 2};
    MPI_Aint rdisps[] = {(sny + 1) * (snx + 2) + 1, 1, snx + 2, 2 * snx + 3};
    MPI_Datatype types[] = {row, row, col, col};
    uint32_t nc = 0;
    for (uint32_t t = 0; t < iters; t++) {
        MPI_Request req;
        MPI_Ineighbor_alltoallw(raw_sub, counts, sdisps, types, raw_sub, counts,
                                rdisps, types, comm, &req);
#pragma omp parallel for
        for (uint32_t r = 1; r <= sny; r++) {
            for (uint32_t c = 1; c < snx; c++) {
                data_sub[r][c] =
                    data_sub[r][c] +
                    a * dt *
                        ((data_sub[r + 1][c] - 2.0 * data_sub[r][c] + data_sub[r - 1][c]) *
                             inv_dx2 +
                         (data_sub[r][c + 1] - 2.0 * data_sub[r][c] + data_sub[r][c - 1]) *
                             inv_dy2);
            }
        }
        MPI_Status stat;
        MPI_Wait(&req, &stat);
        #pragma omp parallel
        {
            #pragma omp task
        {
            for (uint32_t i = 1; i <= snx; i++) {
                data_sub[1][i] = 
                    data_sub[1][i] +
                    a * dt *
                        ((data_sub[1 + 1][i] - 2.0 * data_sub[1][i] + data_sub[1 - 1][i]) *
                             inv_dx2 +
                         (data_sub[1][i + 1] - 2.0 * data_sub[1][i] + data_sub[1][i - 1]) *
                             inv_dy2);
            }
        }
#pragma omp task
        {
            for (uint32_t i = 1; i <= snx; i++) {
                data_sub[sny][i] = 
                    data_sub[sny][i] +
                    a * dt *
                        ((data_sub[sny + 1][i] - 2.0 * data_sub[sny][i] + data_sub[sny - 1][i]) *
                             inv_dx2 +
                         (data_sub[sny][i + 1] - 2.0 * data_sub[sny][i] + data_sub[sny][i - 1]) *
                             inv_dy2);
            }
        }
#pragma omp task
        {
            for (uint32_t i = 1; i <= sny; i++) {
                data_sub[i][1] = 
                    data_sub[i][1] +
                    a * dt *
                        ((data_sub[i + 1][1] - 2.0 * data_sub[i][1] + data_sub[i - 1][1]) *
                             inv_dx2 +
                         (data_sub[i][1 + 1] - 2.0 * data_sub[i][1] + data_sub[i][1 - 1]) *
                             inv_dy2);
            }
        }
#pragma omp task
        {
            for (uint32_t i = 1; i <= sny; i++) {
                data_sub[i][snx + 1] = 
                    data_sub[i][sny] +
                    a * dt *
                        ((data_sub[i + 1][sny] - 2.0 * data_sub[i][sny] + data_sub[i - 1][sny]) *
                             inv_dx2 +
                         (data_sub[i][sny + 1] - 2.0 * data_sub[i][sny] + data_sub[i][sny - 1]) *
                             inv_dy2);
            }
        }
        }
        if (nc == t) {
            MPI_Gatherv(raw_sub, 1, sub_block, raw, counts, raw_offsets, raw_block, 0, comm);
            if (rank == 0) {
                write(raw, t);
            }
            nc += stepping;
            MPI_Barrier(comm);
        }
    }
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

    std::cout << "Got " << ny << " x " << nx << " input array\n"
              << "Iterations: " << iters << "\n"
              << "Stepping: " << stepping << std::endl;

    meta_and_sub(raw);

    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::nanoseconds;

    auto start = high_resolution_clock::now();
    simulate(raw);
    auto end = high_resolution_clock::now();
    double time = duration_cast<nanoseconds>(end - start).count();
    std::cout << "Simulation took " << time / 1000000 << "ms" << std::endl;
    MPI_Finalize();
    return 0;
}
