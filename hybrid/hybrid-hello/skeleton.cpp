#include <iostream>
#include <mpi.h>
#include <omp.h>

int main(int argc, char *argv[])
{
    int my_id;
    int provided, required=MPI_THREAD_FUNNELED;

    MPI_Init_thread(&argc, &argv, required, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

#pragma omp parallel
    {
        int r = omp_get_thread_num();
        std::cout << "HELLO I'M " << my_id << ":" << r << std::endl;
    }

    MPI_Finalize();
    return 0;
}
