#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int rank;
    int array[8][8];

    // Declare a variable storing the MPI datatype
    MPI_Datatype custom;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Initialize arrays
    if (rank == 0) {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                array[i][j] = (i + 1) * 10 + j + 1;
            }
        }
    } else {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                array[i][j] = 0;
            }
        }
    }

    // Print data on rank 0
    if (rank == 0) {
        printf("Data on rank %d\n", rank);
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                printf("%3d", array[i][j]);
            }
            printf("\n");
        }
    }

    // Create datatype

    // Second column
    /*int offsets[8], block_lens[8];
    for (int i = 0; i < 8; i++) {
        offsets[i] = 1 + 8 * i;
        block_lens[i] = 1;
    }
    MPI_Type_indexed(8, block_lens, offsets, MPI_INT, &custom);*/

    // Weird stairstep
    /*int offsets[4], block_lens[4];
    for (int i = 0; i < 4; i++) {
        offsets[i] = 2 * 8 * i + i;
        block_lens[i] = i + 1;
    }
    MPI_Type_indexed(4, block_lens, offsets, MPI_INT, &custom);*/

    int offsets[4], block_lens[4];
    for (int i = 0; i < 4; i++) {
        offsets[i] = 2 * 8 + 2 + i * 8;
        block_lens[i] = 4;
    }
    MPI_Type_indexed(4, block_lens, offsets, MPI_INT, &custom);
    MPI_Type_commit(&custom);

    // Send data from rank 0 to rank 1
    if (rank == 0) {
        MPI_Send(array, 1, custom, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        MPI_Status stats;
        MPI_Recv(array, 1, custom, 0, 0, MPI_COMM_WORLD, &stats);
    }

    // Free datatype
    MPI_Type_free(&custom);

    // Print received data
    if (rank == 1) {
        printf("Received data on rank %d\n", rank);
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                printf("%3d", array[i][j]);
            }
            printf("\n");
        }
    }

    MPI_Finalize();

    return 0;
}
