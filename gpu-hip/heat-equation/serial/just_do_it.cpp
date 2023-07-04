
#include <hip/hip_runtime.h>

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
static double* A_;
static double* B_;

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::nanoseconds;

__global__ void heat(double* memA, double* memB, uint32_t nx) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x + nx;
    memB[i] =
        memA[i] + a * dt *
                      ((memA[i + nx] - 2.0 * memA[i] + memA[i - nx]) * inv_dx2 +
                       (memA[i + 1] - 2.0 * memA[i] + memA[i - 1]) * inv_dy2);
}

double write(double* data, uint32_t t) {
    auto start = high_resolution_clock::now();
    std::ostringstream filename_stream;
    filename_stream << "heat_" << std::setw(4) << std::setfill('0') << t
                    << ".png";
    std::string filename = filename_stream.str();
    save_png(data, ny, nx, filename.c_str(), 'c');
    auto end = high_resolution_clock::now();
    return duration_cast<nanoseconds>(end - start).count();
}

void read_file(char const* fname) {
    std::ifstream in(fname);
    char risuaita;
    in >> risuaita >> nx >> ny;
    rawA = (double*)malloc(nx * ny * sizeof(double));
    for (uint32_t i = 0; i < nx * ny; i++) {
        in >> rawA[i];
    }
}

void defdef() {
    uint64_t size = nx * ny;
    rawA = (double*)malloc(size * sizeof(double));
    uint32_t cut = nx / 10;
    std::fill(rawA, rawA + nx * ny, double(95));
    for (uint32_t r = cut + 1; r < ny - cut - 1; r++) {
        std::fill(rawA + r * nx + (nx / 2 - cut),
                  rawA + r * nx + (nx / 2 + cut), double(0));
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

    hipMalloc((void **) &A_, nx * ny * sizeof(double));
    hipMalloc((void **) &B_, nx * ny * sizeof(double));
    hipMemcpy(A_, rawA, sizeof(double) * nx * ny, hipMemcpyHostToDevice);

    dim3 blocks(2 * (ny - 2));
    dim3 threads(nx / 2);

    double png_time = 0;

    auto start = high_resolution_clock::now();

    uint32_t nc = 0;
    for (uint32_t t = 0; t < iters; t++) {
        heat<<<blocks, threads, 0, 0>>>(A_, B_, nx);
        if (nc == t) {
            hipMemcpy(rawA, B_, sizeof(double) * nx * ny, hipMemcpyDeviceToHost);
            png_time += write(rawA, t);
            nc += stepping;
        }

        std::swap(A_, B_);
    }

    auto end = high_resolution_clock::now();
    double time = duration_cast<nanoseconds>(end - start).count();
    std::cout << "Simulation took " << time / 1000000 << "ms" << std::endl;
    std::cout << "Writing PNG files was " << png_time / 1000000 << "ms of that" << std::endl;
    free(rawA);
    hipFree((void*)A_);
    hipFree((void*)B_);
    return 0;
}
