
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
static uint32_t nx = 20;
static uint32_t ny = nx;
static uint32_t iters = 20;
static uint32_t stepping = 1;
static double* rawA;
static double* rawB;

void write(double* data, uint32_t t) {
    std::ostringstream filename_stream;
    filename_stream << "heat_" << std::setw(4) << std::setfill('0') << t
                    << ".png";
    std::string filename = filename_stream.str();
    save_png(data, ny + 2, nx + 2, filename.c_str(), 'c');
}

void read_file(char const* fname) {
    std::ifstream in(fname);
    char risuaita;
    in >> risuaita >> nx >> ny;
    rawA = (double*)malloc(nx * ny * sizeof(double));
    rawB = (double*)malloc(nx * ny * sizeof(double));
    for (uint32_t r = 0; r < ny; r++) {
        for (uint32_t c = 0; c < nx; c++) {
            in >> rawA[r * nx + c];
        }
    }
}

void defdef() {
    uint64_t size = nx * ny;
    rawA = (double*)malloc(size * sizeof(double));
    rawB = (double*)malloc(size * sizeof(double));
    uint32_t cut = nx / 10;
    std::fill(rawA, rawA + nx * ny, double(95));
    for (uint32_t r = cut + 1; r < ny - cut - 1; r++) {
        std::fill(rawA + r * nx + (nx / 2 - cut), rawA + r * nx + (nx / 2 + cut),
                  double(0));
    }
}

int main(int argc, char* argv[]) {
    if (argc > 1) {
        read_file(argv[1]);
    } else {
        defdef();
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


    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::nanoseconds;

    auto start = high_resolution_clock::now();

    uint32_t nc = 0;
    for (uint32_t t = 0; t < iters; t++) {
        #pragma omp target teams distribute parallel for map(to:rawA[nx * ny],from:rawB[nx * ny])
        for (uint32_t r = 1; r < ny - 1; r++) {
            for (uint32_t c = 1; c < nx - 1; c++) {
                rawB[r * nx + c] =
                    rawA[r * nx + c] +
                    a * dt *
                        ((rawA[(r + 1) * nx + c] - 2.0 * rawA[r * nx + c] +
                          rawA[(r - 1) * nx + c]) *
                             inv_dx2 +
                         (rawA[r * nx + c + 1] - 2.0 * rawA[r * nx + c] +
                          rawA[r * nx + (c - 1)]) *
                             inv_dy2);
            }
        }

        if (nc == t) {
            write(rawB, t);
            nc += stepping;
        }
        std::swap(rawA, rawB);
    }
    
    auto end = high_resolution_clock::now();
    double time = duration_cast<nanoseconds>(end - start).count();
    std::cout << "Simulation took " << time / 1000000 << "ms" << std::endl;
    free(rawA);
    free(rawB);
    return 0;
}
