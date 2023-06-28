#include <iostream>
#include <mpi.h>
#include <cstdlib>

int main(int argc, char *argv[])
{
    int rc = MPI_Init(&argc, &argv);
    int rank, size;
    rc |= MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    rc |= MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rc != MPI_SUCCESS) {
        std::cerr << "Failed to init MPI" << std::endl;
        exit(1);
    }
    

    char* c = (char*)malloc(MPI_MAX_PROCESSOR_NAME);
    int len;
    MPI_Get_processor_name(c, &len);
    

    std::cout << "Hello, I am " << rank << "! Running on " << c << std::endl;
    if (rank == 0) {
        std::cout << "There are a total of " << size << " task" << std::endl;
    }

    MPI_Finalize();
}
