#include "generateRR.h"
#include <mpi.h>
#include <stdio.h>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get my rank id
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    graph mygraph;

    mygraph.nodes.push_back(rank);
    mygraph.nodes.push_back(world_size + rank);
    mygraph.nodes.push_back(2 * world_size + rank);
    mygraph.nodes.push_back(3 * world_size + rank);

    std::cout << "Rank " << rank << " of " << world_size << " have graph of: ";
    for (int i: mygraph.nodes)
    {
        std::cout << i << " ";
    }
    std::cout << "\n";
    generate_RR(mygraph, 2, rank, world_size);

    MPI_Finalize();
    return 0;
}