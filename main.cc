// to compile: mpic++  -std=c++17 main.cc rand_gen.cc
// run example:  mpirun --bind-to core -np 8 /gpfs/u/home/PCPF/PCPFrttn/scratch/proj/self/a.out
#include "generateRR.h"
#include "parsefile.h"
#include "selectSeed.h"

#include <mpi.h>
#include <stdio.h>
#include <iostream>

bool DEBUG_MODE = true;

void populate_graph_by_file(graph &mygraph, const char* fileName)
{
    mygraph.adj_vector = readFile(fileName);
}


int main(int argc, char** argv) {

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>\n";
        return 1;
    }

    MPI_Init(&argc, &argv);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get my rank id
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    graph mygraph;

    const char* filename = argv[1];
    populate_graph_by_file(mygraph, filename);


    if (DEBUG_MODE) {

        std::cout << "Rank " << rank << " of " << world_size << " have graph of: ";
        for (int i = 0; i < mygraph.size(); i++)
        {
            std::cout << id_local_to_global(i, world_size, rank) << " ";
        }
        std::cout << "\n";
        std::cout << "with adjacency matrix\n";
        for (const auto& row : mygraph.adj_vector) {
            for (const auto& elem : row) {
                std::cout << elem << " ";
            }
            std::cout << std::endl;
        }
    }

    unsigned long num_sample = 2;
    std::vector<std::set<NumberType>> RRset;
    RRset = generate_RR(mygraph, num_sample, rank, world_size, DEBUG_MODE);
    

    // not sure if we need to invert the RR set
    std::vector<std::set<NumberType>> emplicitRR;
    emplicitRR = invertNodeWalks(RRset, num_sample * world_size);

    std::vector<std::unordered_set<int>> combined_RR;
    NumberType num_node;
    combined_RR = allrank_combineRR(emplicitRR, rank, world_size, num_node);

    std::vector<std::unordered_set<int>> explicitRR_distributed;
    distribute_walks_cyclic(&combined_RR, explicitRR_distributed, rank, world_size);

    std::vector<NumberType> k_influential;
    int k = 5;

    k_influential = selectSeed2D(explicitRR_distributed, k, num_node, rank, world_size);

    MPI_Finalize();
    return 0;
}
