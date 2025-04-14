// to compile: mpic++  -std=c++11 test.cc rand_gen.cc
// run example:  mpirun --bind-to core -np 8 /gpfs/u/home/PCPF/PCPFrttn/scratch/proj/self/a.out
#include "EdgeInfo.h"
#include "generateRR.h"
#include "parsefile.h"
#include <mpi.h>
#include <iostream>

void populate_graph(graph &mygraph, 
                    int num_node_per_rank, 
                    int num_in_neighbor, 
                    int num_out_neighbor, 
                    int world_size, 
                    int myrank)
{
    NumberType max_id = world_size * num_node_per_rank;

    for (int i = 0; i < num_node_per_rank; i++) {
        std::vector<EdgeType> neighbor_vector;
        NumberType current_id = i * world_size + myrank;

        // add neighbor from remote graph
        for (int k = 0; k < num_out_neighbor; k++) {
            NumberType neighbor_id = (current_id + k + 1) % max_id;
            while ((neighbor_id - myrank) % world_size == 0) {
                neighbor_id++;

                if (neighbor_id >= max_id )
                {
                    neighbor_id = neighbor_id % max_id;
                }
            }
            neighbor_vector.push_back({neighbor_id,1.0});
        }

        // add neighbor from local graph
        for (int j = 0; j < num_in_neighbor; j++) {
            neighbor_vector.push_back({(current_id + (j + 1) * world_size) % max_id,1.0});
        }

        mygraph.adj_vector.push_back(neighbor_vector);
    }
}

void populate_graph(graph &mygraph, const char* fileName)
{
    mygraph.adj_vector = readFile(fileName);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get my rank id
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    graph mygraph;

    // mygraph.nodes.push_back(rank);
    // mygraph.nodes.push_back(world_size + rank);
    // mygraph.nodes.push_back(2 * world_size + rank);
    // mygraph.nodes.push_back(3 * world_size + rank);
    #ifdef GENERATEGRAPH
        populate_graph(mygraph,
                    5,
                    3,
                    2,
                    world_size,
                    rank
    );
    #else
    populate_graph(mygraph,"../edges.txt");
    #endif

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
    generate_RR(mygraph, 2, rank, world_size, true);

    MPI_Finalize();
    return 0;
}