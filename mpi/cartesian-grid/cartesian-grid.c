#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>


int main(int argc, char* argv[]) {
    int ntasks, myid, irank;
    int dims[2] = {0};      /* Dimensions of the grid */
    int coords[2] = {0};    /* Coordinates in the grid */
    int neighbors[4] = {0}; /* Neighbors in 2D grid */
    int period[2] = {1, 1};
    MPI_Comm comm2d;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    /* Determine the process grid (dims[0] x dims[1] = ntasks) */
    if (ntasks < 16) {
        dims[0] = 2;
    } else if (ntasks >= 16 && ntasks < 64) {
        dims[0] = 4;
    } else if (ntasks >= 64 && ntasks < 256) {
        dims[0] = 8;
    } else {
        dims[0] = 16;
    }
    dims[1] = ntasks / dims[0];

    if (dims[0] * dims[1] != ntasks) {
        fprintf(stderr, "Incompatible dimensions: %i x %i != %i\n",
                dims[0], dims[1], ntasks);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    /* Create the 2D Cartesian communicator */
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, period, 0, &cart);
    MPI_Comm_rank(cart, &irank);

    MPI_Cart_coords(cart, irank, 2, coords);
    for (int i = 0; i < 4; i++) {
        int nc[2];
        nc[0] = coords[0] + (i % 2) * (i - 2);
        nc[1] = coords[1] + ((i + 1) % 2) * (i - 1);
        MPI_Cart_rank(cart, nc, neighbors + i);
    }
    

    for (irank = 0; irank < ntasks; irank++) {
        if (myid == irank) {
            printf("%3i = %2i %2i neighbors=%3i %3i %3i %3i\n",
                   myid, coords[0], coords[1], neighbors[0], neighbors[1],
                   neighbors[2], neighbors[3]);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
