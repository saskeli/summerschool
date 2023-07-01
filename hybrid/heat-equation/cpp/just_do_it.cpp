#include <cstdint>
#include <fstream>
#include <string>
#include <iostream>
#include <iomanip> 
#include <chrono>
#include "pngwriter.h"

void write(double* data, uint32_t t, uint32_t nx, uint32_t ny) {
    std::ostringstream filename_stream;
    filename_stream << "heat_" << std::setw(4) << std::setfill('0') << t << ".png";
    std::string filename = filename_stream.str();
    save_png(data, ny, nx, filename.c_str(), 'c');
}

double* read_file(char const* fname, uint32_t& nx, uint32_t& ny) {
    std::ifstream in(fname);
    char c;
    in >> c >> nx >> ny;
    double* raw = (double*)malloc(nx * ny * sizeof(double));
    for (uint32_t i = 0; i < nx * ny; i++) {
        in >> raw[i];
    }
    return raw;
}

double* defdef(uint32_t nx, uint32_t ny) {
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

int main(int argc, char const *argv[])
{
    uint32_t nx = 200;
    uint32_t ny = 200;
    uint32_t iters = 600;
    uint32_t stepping = 100;
    double* raw;
    if (argc > 1) {
        raw = read_file(argv[1], nx, ny);
    } else {
        raw = defdef(nx, ny);
    }
    if (argc > 2) {
        iters = atoll(argv[2]);
    }
    if (argc > 3) {
        stepping = atoll(argv[3]);
    }
    double** data = (double**)malloc(ny * sizeof(double*));
    for (uint32_t i = 0; i < ny; i++) {
        data[i] = raw + i * nx;
    }
    const double a = 0.5;
    const double dx = 0.01;
    const double dy = 0.01;
    const double dt = dx * dx * dy * dy / (2.0 * a * (dx * dx + dy * dy));
    const double inv_dx2 = 1.0 / (dx * dx);
    const double inv_dy2 = 1.0 / (dy * dy);

    using std::chrono::duration_cast;
    using std::chrono::high_resolution_clock;
    using std::chrono::nanoseconds;

    auto start = high_resolution_clock::now();
    uint32_t nc = 0;
    for (uint32_t t = 0; t < iters; t++) {
        for (uint32_t r = 1; r < ny - 1; r++) {
            for (uint32_t c = 1; c < nx - 1; c++) {
                data[r][c] = data[r][c] + a * dt *
                    ((data[r + 1][c] - 2.0 * data[r][c] + data[r - 1][c]) * inv_dx2 +
                     (data[r][c + 1] - 2.0 * data[r][c] + data[r][c - 1]) * inv_dy2);
            }
        }
        if (nc == t) {
            write(raw, t, nx, ny);
            nc += stepping;
        }
    }
    auto end = high_resolution_clock::now();
    double time = duration_cast<nanoseconds>(end - start).count();
    std::cout << "Simulation took " << time / 1000000 << "ms" << std::endl;
    return 0;
}
