#include <mpi.h>
#include <iostream>
#include <cmath>
#include <cstdio>

constexpr int n = 1000000;

int main(int argc, char** argv) {
    int rc, rank, size;
    rc = MPI_Init(&argc, &argv);
    rc |= MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    rc |= MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rc != MPI_SUCCESS) {
        std::cerr << "I'm broken" << std::endl;
        exit(1);
    }

    if (rank == 0) printf("Computing approximation to pi with N=%d\n", n);

    int an_big_steppy = n / size;
    int istart = rank * an_big_steppy;
    int istop = rank == size - 1 ? n + 1 : istart + an_big_steppy;

    double pi = 0.0;
    for (int i = istart; i < istop; i++) {
        double x = (i - 0.5) / n;
        pi += 1.0 / (1.0 + x * x);
    }
    double poo;
    MPI_Reduce(&pi, &poo, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        pi = poo * 4.0 / n;
        printf("Approximate pi=%18.16f (exact pi=%10.8f)\n", pi, M_PI);
    }
    MPI_Finalize();
}
