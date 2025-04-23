#include <mpi.h>
#include <vector>
#include <set>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include "EdgeInfo.h"
#include "util.h"

typedef EdgeType::IntType NumberType;


inline std::vector<NumberType> selectSeed2D(std::vector<std::unordered_set<NumberType>> R, int k, NumberType num_node, int myrank, int world_size, bool DEBUG_MODE)
{
	std::vector<NumberType> selected_nodes; // Output set
	selected_nodes.reserve(k);
	std::vector<NumberType> local_count(num_node / world_size + 1, 0); // Count array
	std::vector<std::vector<NumberType>> local_C(num_node, std::vector<NumberType>(num_node / world_size + 1));

	std::unordered_map<int, std::vector<NumberType>> count_buffers;
	std::unordered_map<int, std::vector<NumberType>> cooccur_buffers;

	if (DEBUG_MODE)
	{
        std::cout << "Rank [" << myrank << "]" << " SelectSeed2D: preprocessing, num_node = " << num_node << "\n";
	}
	// Preprocessing step
	size_t R_size = R.size();
	for (int i = 0; i < R_size; i++)
	{
		// A single RRR set
		std::unordered_set<NumberType> &T = R[i]; // reference

		// Build matrix
		std::vector<NumberType> T_nodes(T.begin(), T.end()); // all number in set
		for (int j = 0; j < T_nodes.size(); j++)
		{
			// Count nodes in current RRR set
			int destination = T_nodes[j] % world_size;
			add_count_batch(T_nodes[j] , myrank, world_size, count_buffers, local_count);
			
			for (int l = j; l < T_nodes.size(); l++)
			{
				NumberType a = T_nodes[j];
				NumberType b = T_nodes[l];
				
				// Increment node pair co-occurences in current set
				add_occurance_batch(T_nodes[j], T_nodes[l], myrank, world_size, cooccur_buffers, local_C);
				if (a != b)
					add_occurance_batch(T_nodes[l], T_nodes[j], myrank, world_size, cooccur_buffers, local_C);
			}
		}
	}
	
	// mpi barrier here
	MPI_Barrier(MPI_COMM_WORLD);
	if (DEBUG_MODE)
	{
		std::cout << "Rank [" << myrank << "]" << " SelectSeed2D: count message sending phase\n";
	}
	flush_count_messages(count_buffers, myrank, world_size, get_mpi_type<NumberType>());
	if (DEBUG_MODE)
	{
		std::cout << "Rank [" << myrank << "]" << " SelectSeed2D: count message receiving phase\n";
	}
	receive_count_messages(local_count, myrank, world_size, get_mpi_type<NumberType>());
	
	MPI_Barrier(MPI_COMM_WORLD);


	// now receive updates
	if (DEBUG_MODE)
	{
		std::cout << "Rank [" << myrank << "]" << " SelectSeed2D: C matrix message sending phase\n";
	}
	flush_occurance_messages(cooccur_buffers, myrank, world_size, get_mpi_type<NumberType>());
	if (DEBUG_MODE)
	{
		std::cout << "Rank [" << myrank << "]" << " SelectSeed2D: C matrix message receiving phase\n";
	}
	receive_occurance_messages(local_C, myrank, world_size, get_mpi_type<NumberType>());

	// Seed Selection
	MPI_Datatype mpi2num_type = create_MPI_2NUM();
	MPI_Op maxloc_op;
	MPI_Op_create(&maxloc_reduce, /*commute=*/1, &maxloc_op);

	for (int i = 0; i < k; i++)
	{
		// Global argmax
		NumberType global_max_node = find_global_argmax(local_count, myrank, world_size, mpi2num_type, maxloc_op);
		selected_nodes.push_back(global_max_node);

		// If this rank owns global_max_node, zero out its count
		if (global_max_node % world_size == myrank)
		{
			NumberType local_index = global_max_node / world_size;
			if (local_index >= 0 && local_index < static_cast<NumberType>(local_count.size()))
				local_count[local_index] = 0;
		}

		// subtract influence of global_max_node from all ranks
		for (NumberType i = 0; i < local_count.size(); ++i)
		{
			NumberType global_col = i * world_size + myrank;
			local_count[i] -= local_C[global_max_node][i]; // subtract count by row C[global_max_node] 
			if (local_count[i] < 0)
				local_count[i] = 0;
		}

		MPI_Barrier(MPI_COMM_WORLD);
	}

	MPI_Op_free(&maxloc_op);
	MPI_Type_free(&mpi2num_type);
	return selected_nodes;
}
