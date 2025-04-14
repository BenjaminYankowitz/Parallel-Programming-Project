// to compile: mpic++  -std=c++11 main.cc rand_gen.cc
// run example:  mpirun --bind-to core -np 8 /gpfs/u/home/PCPF/PCPFrttn/scratch/proj/self/a.out
#include "generateRR.h"
#include "parsefile.h"
#include <mpi.h>
#include <stdio.h>
#include <iostream>

bool DEBUG_MODE = true;

void populate_graph_by_file(graph &mygraph, const char* fileName)
{
    mygraph.adj_vector = readFile(fileName);
}

// TODO: import actual selectSeed
std::vector<std::set<NumberType>> selectSeed (std::vector<std::set<NumberType>> RRset)
{
    std::set<NumberType>
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


    MPI_Finalize();
    return 0;
}