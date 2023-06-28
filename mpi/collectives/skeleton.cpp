#include <cstdio>
#include <vector>
#include <cstring>
#include <mpi.h>

#define NTASKS 4

void init_buffers(std::vector<int> &sendbuffer);
void print_buffers(std::vector<int> &buffer);


int main(int argc, char *argv[])
{
    int ntasks, rank;
    std::vector<int> sendbuf(2 * NTASKS);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (ntasks != NTASKS) {
        if (rank == 0) {
            fprintf(stderr, "Run this program with %i tasks.\n", NTASKS);
        }
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    /* Initialize message buffers */
    init_buffers(sendbuf);
    print_buffers(sendbuf);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(sendbuf.data(), sendbuf.size(), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    print_buffers(sendbuf);
    init_buffers(sendbuf);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Scatter(sendbuf.data(), sendbuf.size() / NTASKS, MPI_INT, rank == 0 ? MPI_IN_PLACE : sendbuf.data(), sendbuf.size() / NTASKS, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    print_buffers(sendbuf);
    init_buffers(sendbuf);
    MPI_Barrier(MPI_COMM_WORLD);

    int arr[] = {1, 1, 2, 4};
    int arrb[] = {0, 1, 2, 4};
    int* bla = (int*)malloc(2 * NTASKS * sizeof(int));
    std::memcpy(bla, sendbuf.data(), 2 * NTASKS * sizeof(int));

    MPI_Gatherv(bla, sendbuf.size(), MPI_INT, sendbuf.data(), arr, arrb, MPI_INT, 1, MPI_COMM_WORLD);
    print_buffers(sendbuf);
    init_buffers(sendbuf);
    MPI_Barrier(MPI_COMM_WORLD);
    
    MPI_Finalize();
    return 0;
}


void init_buffers(std::vector<int> &sendbuffer)
{
    int rank;
    int buffersize = sendbuffer.size();

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (int i = 0; i < buffersize; i++) {
        sendbuffer[i] = i + buffersize * rank;
    }
}


void print_buffers(std::vector<int> &buffer)
{
    int rank, ntasks;
    int buffersize = buffer.size();

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

    std::vector<int> printbuffer(buffersize * ntasks);

    MPI_Gather(buffer.data(), buffersize, MPI_INT,
               printbuffer.data(), buffersize, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int j = 0; j < ntasks; j++) {
            printf("Task %2i:", j);
            for (int i = 0; i < buffersize; i++) {
                printf(" %2i", printbuffer[i + buffersize * j]);
            }
            printf("\n");
        }
        printf("\n");
    }
}
