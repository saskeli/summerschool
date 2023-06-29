#include <cstdio>
#include <vector>
#include <mpi.h>

void print_ordered(double t);

int main(int argc, char *argv[])
{
    int i, myid, ntasks;
    int ONE = 1;
    constexpr int size = 10000000;
    std::vector<int> message(size);
    std::vector<int> receiveBuffer(size);
    MPI_Status status;

    double t0, t1;

    int source, destination;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    // Initialize buffers
    for (i = 0; i < size; i++) {
        message[i] = myid;
        receiveBuffer[i] = -1;
    }

    int to, from;

    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 1, &ntasks, &ONE, 0, &cart);

    MPI_Cart_shift(cart, 0, 1, &from, &to); 

    // Start measuring the time spent in communication
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    MPI_Request reqs[2];
    MPI_Status stats[2];
    MPI_Send_init(message.data(), size, MPI_INT, to, 0, MPI_COMM_WORLD, reqs);

    printf("Sender: %d. Sent elements: %d. Tag: %d. Receiver: %d\n",
           myid, size, myid + 1, destination);

    MPI_Recv_init(receiveBuffer.data(), size, MPI_INT, from, 0, MPI_COMM_WORLD, reqs + 1);

    MPI_Startall(2, reqs);
    MPI_Waitall(2, reqs, stats);

    printf("Receiver: %d. first element %d.\n",
           myid, receiveBuffer[0]);

    // Finalize measuring the time and print it out
    t1 = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    fflush(stdout);

    print_ordered(t1 - t0);

    MPI_Finalize();
    return 0;
}

void print_ordered(double t)
{
    int i, rank, ntasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

    if (rank == 0) {
        printf("Time elapsed in rank %2d: %6.3f\n", rank, t);
        for (i = 1; i < ntasks; i++) {
            MPI_Recv(&t, 1, MPI_DOUBLE, i, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Time elapsed in rank %2d: %6.3f\n", i, t);
        }
    } else {
        MPI_Send(&t, 1, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD);
    }
}
