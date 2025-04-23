// to compile: mpic++  -std=c++17 main.cc rand_gen.cc
// run example:  mpirun --bind-to core -np 8 /gpfs/u/home/PCPF/PCPFrttn/scratch/proj/self/a.out
#include "generateRR.h"
#include "parsefile.h"
#include "selectSeed.h"
#include "clockcycle.h"

#include <mpi.h>
#include <iostream>

bool DEBUG_MODE = false;

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
	
	NumberType global_max_id = find_global_max_node_id(
        mygraph.size(),
        world_size,
        rank,
        id_local_to_global);
	
    unsigned long num_sample = (global_max_id + 1) / world_size; // num walks
    std::vector<std::set<NumberType>> RRset;
    ticks startGenRRTimer = clock_now();
    RRset = generate_RR(mygraph, num_sample, rank, world_size, DEBUG_MODE);
    ticks endGenRRTimer = clock_now();
	
	ticks startSetupSSTimer = clock_now();
    // not sure if we need to invert the RR set
    if (DEBUG_MODE) { std::cout << "inverting RRR set\n"; }
    std::vector<std::set<NumberType>> emplicitRR;
    emplicitRR = invertNodeWalks(RRset, num_sample * world_size);
    
    if (DEBUG_MODE) { std::cout << "collectively combining RRR set\n"; }
    std::vector<std::unordered_set<int>> combined_RR;
    NumberType num_node;
    combined_RR = allrank_combineRR(emplicitRR, rank, world_size, num_node);

	if (DEBUG_MODE) { std::cout << "distributing RRR set\n"; }
    std::vector<std::unordered_set<int>> explicitRR_distributed;
    distribute_walks_cyclic(&combined_RR, explicitRR_distributed, rank, world_size);
	ticks endSetupSSTimer = clock_now();
    
    std::vector<NumberType> k_influential;
    int k = 5;
	
	if (DEBUG_MODE) { std::cout << "computing selectSeed\n"; }
	ticks startSelectSeedTimer = clock_now();
    k_influential = selectSeed2D(explicitRR_distributed, k, global_max_id + 1, rank, world_size, DEBUG_MODE);
	ticks endSelectSeedTimer = clock_now();
	ticks endOverallTimer = clock_now();
	
	if (DEBUG_MODE)
    {
        if (rank == 0)
        {
            
        }
    }
	
    MPI_Finalize();
    if (rank == 0)
    {
		std::cerr << "Elapsed time (Reading in data) = " 
		<< getCycles(startIOTimer, endIOTimer) << " ticks, " 
		<< getElapsedSeconds(startIOTimer, endIOTimer) << " seconds\n";
		std::cerr << "Elapsed time (Generating RRR sets) = " 
		<< getCycles(startGenRRTimer, endGenRRTimer) << " ticks, " 
		<< getElapsedSeconds(startGenRRTimer, endGenRRTimer) << " seconds\n";
		std::cerr << "Elapsed time (Setting up RRR sets for seed selection) = " 
		<< getCycles(startSetupSSTimer, endSetupSSTimer) << " ticks, " 
		<< getElapsedSeconds(startSetupSSTimer, endSetupSSTimer) << " seconds\n";
		std::cerr << "Elapsed time (Selecting influential seeds) = " 
		<< getCycles(startSelectSeedTimer, endSelectSeedTimer) << " ticks, " 
		<< getElapsedSeconds(startSelectSeedTimer, endSelectSeedTimer) << " seconds\n";
		std::cerr << "Elapsed time (Overall time taken) = " 
		<< getCycles(startOverallTimer, endOverallTimer) << " ticks, " 
		<< getElapsedSeconds(startOverallTimer, endOverallTimer) << " seconds\n";
		
		std::cerr << "Most influential node: ";
        for (int i = 0; i < k_influential.size(); i++)
        {
            std::cerr << k_influential[i] << " ";
        }
        std::cerr << "\n";	
		//std::cout << "Elapsed time (Reading in data) = " << getElapsedSeconds(startIOTimer, endIOTimer) << " seconds\n";
			

		//printf("Total samples = %lld\n", total_samples);
		//printf("Approximate cycles = %.0f cycles\n", cycles);
    }
    return 0;
}
