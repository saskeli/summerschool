#include <cstdio>
#include <vector>
#include <mpi.h>
#include <iostream>
#include <deque>
#include <unordered_set>

void init_buffers(std::vector<int> &sendbuffer);
void print_buffers(std::vector<int> &buffer);


int main(int argc, char *argv[])
{
    int ntasks, myid, size=12;
    std::vector<int> sendbuf(size);
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if (ntasks < 2) {
        std::cerr << "Too few tasks" << std::endl;
        exit(1);
    }

    /* Initialize message buffers */
    init_buffers(sendbuf);

    /* Print data that will be sent */
    print_buffers(sendbuf);
    MPI_Barrier(MPI_COMM_WORLD);
    /* Send everywhere */
    std::deque<int>* a = new std::deque<int>();
    std::deque<int>* b = new std::deque<int>();
    std::unordered_set<int> has;
    has.insert(0);
    a->push_back(0);
    int hop = (ntasks + 1) / 2;
    while (hop) {
        while (a->size()) {
            int src = a->front();
            a->pop_front();
            b->push_back(src);
            int trg = (src + hop) % ntasks;
            if (has.count(trg) == 0) {
                b->push_back(trg);
                if (myid == src) {
                    MPI_Send(sendbuf.data(), size, MPI_INT, trg, 0, MPI_COMM_WORLD);
                }
            }
        }
        for (auto trg : *b) {
            if (has.count(trg)) {
                continue;
            }
            has.insert(trg);
            int src = trg - hop;
            if (myid == trg) {
                MPI_Recv(sendbuf.data(), size, MPI_INT, src, 0, MPI_COMM_WORLD, &status);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        std::swap(a, b);
        hop = hop == 1 ? 0 : (hop + 1) / 2;
    }

    print_buffers(sendbuf);

    MPI_Finalize();
    return 0;
}


void init_buffers(std::vector<int> &sendbuffer)
{
    int rank;
    int buffersize = sendbuffer.size();

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        for (int i = 0; i < buffersize; i++) {
            sendbuffer[i] = i;
        }
    } else {
        for (int i = 0; i < buffersize; i++) {
            sendbuffer[i] = -1;
        }
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
