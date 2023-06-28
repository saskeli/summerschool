#include <mpi.h>

#include <cmath>
#include <cstdio>

constexpr int n = 840;

int main(int argc, char** argv) {
    int rc, rank;
    rc = MPI_Init(&argc, &argv);
    rc |= MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) printf("Computing approximation to pi with N=%d\n", n);

    int istart = rank == 0 ? n / 2 : 1;
    int istop = rank == 0 ? n + 1 : n / 2;

    double pi = 0.0;
    for (int i = istart; i < istop; i++) {
        double x = (i - 0.5) / n;
        pi += 1.0 / (1.0 + x * x);
    }

    if (rank == 1) {
        MPI_Send(&pi, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    } else {
        double to_add;
        MPI_Status stat;
        MPI_Recv(&to_add, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &stat);
        pi += to_add;
        pi *= 4.0 / n;
        printf("Approximate pi=%18.16f (exact pi=%10.8f)\n", pi, M_PI);
    }
   
    MPI_Finalize();
}
