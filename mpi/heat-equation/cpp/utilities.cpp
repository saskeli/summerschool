// Utility functions for heat equation solver

#include <mpi.h>

#include "heat.hpp"

// Calculate average temperature
double average(const Field& field, const ParallelData parallel) {
    double local_average = 0.0;
    double average = 0.0;

    for (int i = 1; i < field.nx + 1; i++) {
        for (int j = 1; j < field.ny + 1; j++) {
            local_average += field.temperature(i, j);
        }
    }

    MPI_Reduce(&local_average, &average, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return average / (field.nx_full * field.ny_full);
}
