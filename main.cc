// to compile: mpic++  -std=c++17 main.cc rand_gen.cc
// run example:  mpirun --bind-to core -np 8 /gpfs/u/home/PCPF/PCPFrttn/scratch/proj/self/a.out
#include "generateRR.h"
#include "parsefile.h"
#include "selectSeed.h"
#include "clockcycle.h"

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
    
    ticks startOverallTimer = clock_now();
    
    ticks startIOTimer = clock_now();
    populate_graph_by_file(mygraph, filename);
	ticks endIOTimer = clock_now();

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

	
    unsigned long num_sample = 100; // num walks
    std::vector<std::set<NumberType>> RRset;
    ticks startGenRRTimer = clock_now();
    RRset = generate_RR(mygraph, num_sample, rank, world_size, DEBUG_MODE);
    ticks endGenRRTimer = clock_now();

	//std::cout << "TESTING-------------------------------------------------------------------\n";
	
	ticks startSetupSSTimer = clock_now();
    // not sure if we need to invert the RR set
    std::vector<std::set<NumberType>> emplicitRR;
    emplicitRR = invertNodeWalks(RRset, num_sample * world_size);

    std::vector<std::unordered_set<int>> combined_RR;
    NumberType num_node;
    combined_RR = allrank_combineRR(emplicitRR, rank, world_size, num_node);

    std::vector<std::unordered_set<int>> explicitRR_distributed;
    distribute_walks_cyclic(&combined_RR, explicitRR_distributed, rank, world_size);
	ticks endSetupSSTimer = clock_now();
	//std::cout << "TESTING2-------------------------------------------------------------------\n";
    std::vector<NumberType> k_influential;
    int k = 5;
	
	//std::cout << "TESTING3-------------------------------------------------------------------\n";
	ticks startSelectSeedTimer = clock_now();
    k_influential = selectSeed2D(explicitRR_distributed, k, num_node, rank, world_size);
	ticks endSelectSeedTimer = clock_now();

	//std::cout << "TESTING4 (NOT WORKING)---------------------------------------------------------\n";
	ticks endOverallTimer = clock_now();
	
    MPI_Finalize();
    
    std::cout << "Elapsed time (Reading in data) = " << getElapsedSeconds(startIOTimer, endIOTimer) << " seconds\n";
	std::cout << "Elapsed time (Generating RRR sets) = " << getElapsedSeconds(startGenRRTimer, endGenRRTimer) << " seconds\n";
	std::cout << "Elapsed time (Setting up RRR sets for seed selection) = " << getElapsedSeconds(startSetupSSTimer, endSetupSSTimer) << " seconds\n";
	std::cout << "Elapsed time (Selecting influential seeds) = " << getElapsedSeconds(startSelectSeedTimer, endSelectSeedTimer) << " seconds\n";
	std::cout << "Elapsed time (Overall time taken) = " << getElapsedSeconds(startOverallTimer, endOverallTimer) << " seconds\n";
		
	//std::cout << "Elapsed time (Reading in data) = " << getElapsedSeconds(startIOTimer, endIOTimer) << " seconds\n";
		

	//printf("Total samples = %lld\n", total_samples);
	//printf("Approximate cycles = %.0f cycles\n", cycles);
    
    return 0;
}
