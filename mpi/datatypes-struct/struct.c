#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int n = 1000, reps = 10000;

    typedef struct {
        float coords[3];
        int charge;
        char label[2];
    } particle;

    particle particles[n];

    int i, j, myid;
    double t1, t2;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    // Fill in some values for the particles
    if (myid == 0) {
        for (i = 0; i < n; i++) {
            for (j = 0; j < 3; j++) {
                particles[i].coords[j] = (float)rand() / (float)RAND_MAX * 10.0;
            }
            particles[i].charge = 54;
            strcpy(particles[i].label, "Xe");
        }
    }

    // Define datatype for the struct
    MPI_Datatype custom;
    int sizes[6] = {sizeof(float), sizeof(float), sizeof(float),
                    sizeof(int),   sizeof(char),  sizeof(char)};
    MPI_Aint disps[6];
    disps[0] = 0;
    for (int i = 1; i < 6; i++) {
        disps[i] = disps[i - 1] + sizes[i - 1];
    }
    MPI_Datatype typs[6] = {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT,
                            MPI_INT,   MPI_CHAR,  MPI_CHAR};
    MPI_Type_create_struct(6, sizes, disps, typs, &custom);
    MPI_Datatype Mpart;
    MPI_Type_create_resized(custom, 0, sizeof(particle), &Mpart);
    MPI_Type_commit(&Mpart);

    // Check extent
    MPI_Aint lb, extent;
    MPI_Type_get_extent(custom, &lb, &extent);
    if (myid == 0) {
        printf("Extent before resize is %ld elements vs. %ld\n", extent, sizeof(particle));
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Communicate using the created particletype
    // Multiple sends are done for better timing
    t1 = MPI_Wtime();
    if (myid == 0) {
        for (i = 0; i < reps; i++) {
            MPI_Send(particles, sizeof(particle) * n, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
        }
    } else if (myid == 1) {
        MPI_Status stats;
        for (i = 0; i < reps; i++) {
            MPI_Recv(particles, sizeof(particle) * n, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &stats);
        }
    }
    t2 = MPI_Wtime();

    printf("Time: %i, %e \n", myid, (t2 - t1) / (double)reps);
    printf("Check: %i: %s %f %f %f \n", myid, particles[n - 1].label,
           particles[n - 1].coords[0], particles[n - 1].coords[1],
           particles[n - 1].coords[2]);

    // Free datatype
    MPI_Type_free(&Mpart);

    MPI_Finalize();
    return 0;
}
