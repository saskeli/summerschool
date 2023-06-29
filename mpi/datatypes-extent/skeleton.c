#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int rank, ntasks;
    int array[8][6];

    // Declare a variable storing the MPI datatype
    MPI_Datatype custom;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

    // Initialize arrays
    if (rank == 0) {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 6; j++) {
                array[i][j] = i * 6 + j + 1;
            }
        }
    } else {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 6; j++) {
                array[i][j] = 0;
            }
        }
    }

    // Print data on rank 0
    if (rank == 0) {
        printf("Data on rank %d\n", rank);
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 6; j++) {
                printf("%3d", array[i][j]);
            }
            printf("\n");
        }
    }

    // Create datatype

    // Second column
    MPI_Type_vector(8, 1, 6, MPI_INT, &custom);
    MPI_Datatype col;
    MPI_Type_create_resized(custom, 0, sizeof(int), &col);
    MPI_Type_commit(&col);

    MPI_Scatter(array, 1, col, rank ? array : MPI_IN_PLACE, 1, col, 0, MPI_COMM_WORLD);

    // Free datatype
    MPI_Type_free(&col);

    // Print received data
    for (int i = 0; i < ntasks; i++) {
        if (rank == i) {
            printf("Received data on rank %d\n", rank);
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 6; j++) {
                    printf("%3d", array[i][j]);
                }
                printf("\n");
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    

    MPI_Finalize();

    return 0;
}
