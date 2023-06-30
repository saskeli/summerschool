#pragma once
#include <string>
#include "matrix.hpp"
#include <mpi.h>
#include <iostream>

// Class for basic parallelization information
struct ParallelData {
    MPI_Comm comm;
    MPI_Datatype row, col;
    MPI_Datatype types[4];
    MPI_Aint sdisps[4], rdisps[4];
    int dims[2];
    int rank;
    int size;

    ParallelData() {      // Constructor
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        int periods[] = {0, 0};
        MPI_Dims_create(size, 2, dims);
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm);
        MPI_Comm_rank(comm, &rank);

    };

    void makeTypes(int nx, int ny) {
        nx /= dims[1];
        ny /= dims[0];
        MPI_Type_contiguous(nx, MPI_DOUBLE, &row);
        MPI_Type_vector(ny, 1, nx + 2, MPI_DOUBLE, &col);
        MPI_Type_commit(&row);
        MPI_Type_commit(&col);
        types[0] = row;
        types[1] = row;
        types[2] = col;
        types[3] = col;
        sdisps[0] = ny * (nx + 2) + 1;
        sdisps[1] = nx + 3;
        sdisps[2] = nx + 3;
        sdisps[3] = 2 * nx + 2;
        rdisps[0] = (ny + 1) * (nx + 2) + 1;
        rdisps[1] = 1;
        rdisps[2] = nx + 2;
        rdisps[3] = 2 * nx + 3;
    }

};

// Class for temperature field
struct Field {
    // nx and ny are the true dimensions of the field. The temperature matrix
    // contains also ghost layers, so it will have dimensions nx+2 x ny+2
    int nx;                     // Local dimensions of the field
    int ny;
    int nx_full;                // Global dimensions of the field
    int ny_full;                // Global dimensions of the field
    double dx = 0.01;           // Grid spacing
    double dy = 0.01;

    Matrix<double> temperature;

    void setup(int nx_in, int ny_in, ParallelData& parallel);

    void generate(const ParallelData& parallel);

    // standard (i,j) syntax for setting elements
    double& operator()(int i, int j) {return temperature(i, j);}

    // standard (i,j) syntax for getting elements
    const double& operator()(int i, int j) const {return temperature(i, j);}

};

// Function declarations
void initialize(int argc, char *argv[], Field& current,
                Field& previous, int& nsteps, ParallelData& parallel);

void exchange(Field& field, const ParallelData& parallel, MPI_Request* reqs);

void evolve(Field& curr, const Field& prev, const double a, const double dt);
void evolve_inner(Field& curr, const Field& prev, const double a, const double dt);
void evolve_outer(Field& curr, const Field& prev, const double a, const double dt);

void write_field(const Field& field, const int iter, const ParallelData& parallel);

void read_field(Field& field, std::string filename,
                ParallelData& parallel);

double average(const Field& field, const ParallelData& parallel);
