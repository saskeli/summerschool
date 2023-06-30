// Main solver routines for heat equation solver

#include <mpi.h>

#include "heat.hpp"

static const int counts[] = {1, 1, 1, 1};

// Exchange the boundary values
void exchange(Field& field, const ParallelData& parallel, MPI_Request* reqs) {

    MPI_Ineighbor_alltoallw(field.temperature.data(), counts, parallel.sdisps,
                            parallel.types, field.temperature.data(), counts,
                            parallel.rdisps, parallel.types, parallel.comm,
                            reqs);
}

// Update the temperature values using five-point stencil */
void evolve(Field& curr, const Field& prev, const double a, const double dt) {
    // Compilers do not necessarily optimize division to multiplication, so make
    // it explicit
    auto inv_dx2 = 1.0 / (prev.dx * prev.dx);
    auto inv_dy2 = 1.0 / (prev.dy * prev.dy);

    // Determine the temperature field at next time step
    // As we have fixed boundary conditions, the outermost gridpoints
    // are not updated.
    for (int i = 1; i < curr.nx + 1; i++) {
        for (int j = 1; j < curr.ny + 1; j++) {
            curr(i, j) =
                prev(i, j) +
                a * dt *
                    ((prev(i + 1, j) - 2.0 * prev(i, j) + prev(i - 1, j)) *
                         inv_dx2 +
                     (prev(i, j + 1) - 2.0 * prev(i, j) + prev(i, j - 1)) *
                         inv_dy2);
        }
    }
}

void evolve_inner(Field& curr, const Field& prev, const double a,
                  const double dt) {
    // Compilers do not necessarily optimize division to multiplication, so make
    // it explicit
    auto inv_dx2 = 1.0 / (prev.dx * prev.dx);
    auto inv_dy2 = 1.0 / (prev.dy * prev.dy);

    // Determine the temperature field at next time step
    // As we have fixed boundary conditions, the outermost gridpoints
    // are not updated.
    for (int i = 2; i < curr.nx; i++) {
        for (int j = 2; j < curr.ny; j++) {
            curr(i, j) =
                prev(i, j) +
                a * dt *
                    ((prev(i + 1, j) - 2.0 * prev(i, j) + prev(i - 1, j)) *
                         inv_dx2 +
                     (prev(i, j + 1) - 2.0 * prev(i, j) + prev(i, j - 1)) *
                         inv_dy2);
        }
    }
}

void evolve_outer(Field& curr, const Field& prev, const double a,
                  const double dt) {
    // Compilers do not necessarily optimize division to multiplication, so make
    // it explicit
    auto inv_dx2 = 1.0 / (prev.dx * prev.dx);
    auto inv_dy2 = 1.0 / (prev.dy * prev.dy);

    // Determine the temperature field at next time step
    // As we have fixed boundary conditions, the outermost gridpoints
    // are not updated.
    for (int j = 1; j < curr.ny + 1; j++) {
        curr(1, j) =
            prev(1, j) +
            a * dt *
                ((prev(1 + 1, j) - 2.0 * prev(1, j) + prev(0, j)) * inv_dx2 +
                 (prev(1, j + 1) - 2.0 * prev(1, j) + prev(1, j - 1)) *
                     inv_dy2);
        curr(curr.nx, j) = prev(curr.nx, j) +
                           a * dt *
                               ((prev(curr.nx + 1, j) - 2.0 * prev(curr.nx, j) +
                                 prev(curr.nx - 1, j)) *
                                    inv_dx2 +
                                (prev(curr.nx, j + 1) - 2.0 * prev(curr.nx, j) +
                                 prev(curr.nx, j - 1)) *
                                    inv_dy2);
    }
    for (int i = 2; i < curr.nx; i++) {
        curr(i, 1) = prev(i, 1) +
                     a * dt *
                         ((prev(i + 1, 1) - 2.0 * prev(i, 1) + prev(i - 1, 1)) *
                              inv_dx2 +
                          (prev(i, 1 + 1) - 2.0 * prev(i, 1) + prev(i, 1 - 1)) *
                              inv_dy2);

        curr(i, curr.ny) = prev(i, curr.ny) +
                           a * dt *
                               ((prev(i + 1, curr.ny) - 2.0 * prev(i, curr.ny) +
                                 prev(i - 1, curr.ny)) *
                                    inv_dx2 +
                                (prev(i, curr.ny + 1) - 2.0 * prev(i, curr.ny) +
                                 prev(i, curr.ny - 1)) *
                                    inv_dy2);
    }
}
