#include <mpi.h>
#include <vector>
#include <set>
#include <algorithm>
#include <iostream>
#include <unordered_map>

#include "EdgeInfo.h"
#include "util.h"

typedef EdgeType::IntType NumberType;

// void add_count(std::vector<NumberType> &local_count, NumberType node_index, int myrank, int world_size)
// {
// 	// local_count is N x N/p matrix
// 	// N is total number of node in whole graph, p is number of rank

// 	// check owner
// 	int destination = node_index % world_size;
// 	if (destination == myrank)
// 	{
// 		// this rank own info on this node
// 		local_count[node_index]++;
// 	}
// 	else
// 	{
// 		// send to others
// 		static_assert(std::is_same<NumberType, long>::value || std::is_same<NumberType, int>::value);
// 		if (std::is_same<NumberType, long>::value)
// 		{
// 			// using num type = long
// 			MPI_Send(&node_index, 1, MPI_LONG, destination, 0, MPI_COMM_WORLD); // using tag = 0
// 		}
// 		else if (std::is_same<NumberType, int>::value)
// 		{
// 			// using num type = int
// 			MPI_Send(&node_index, 1, MPI_INT, destination, 0, MPI_COMM_WORLD);
// 		}
// 		else
// 		{
// 			std::cout << "NumberType is not a long or int\n";
// 			exit(-1);
// 		}
// 	}
// }

// void add_occurance(std::vector<std::vector<int>> &local_C, NumberType row_index, NumberType col_index, int myrank, int world_size)
// {
// 	// check owner
// 	int destination = row_index % world_size;
// 	if (destination == myrank)
// 	{
// 		// this rank own info on this row
// 		local_C[row_index][col_index]++;
// 	}
// 	else
// 	{
// 		// send to others
// 		NumberType buffer[2] = {row_index, col_index};
// 		static_assert(std::is_same<NumberType, long>::value || std::is_same<NumberType, int>::value);
// 		if (std::is_same<NumberType, long>::value)
// 		{
// 			// using num type = long
// 			MPI_Send(buffer, 2, MPI_LONG, destination, 0, MPI_COMM_WORLD); // using tag = 0
// 		}
// 		else if (std::is_same<NumberType, int>::value)
// 		{
// 			// using num type = int
// 			MPI_Send(buffer, 2, MPI_INT, destination, 0, MPI_COMM_WORLD);
// 		}
// 		else
// 		{
// 			std::cout << "NumberType is not a long or int\n";
// 			exit(-1);
// 		}
// 	}
// }

std::vector<NumberType> selectSeed2D(std::vector<std::unordered_set<NumberType>> R, int k, NumberType num_node, int myrank, int world_size)
{
	std::vector<NumberType> selected_nodes; // Output set
	selected_nodes.reserve(k);
	std::vector<NumberType> local_count(num_node / world_size + 1, 0); // Count array
	// std::vector<std::vector<NumberType>> C(num_node, std::vector<NumberType>(num_node, 0)); // Co-occurrence matrix
	std::vector<std::vector<NumberType>> local_C(num_node, std::vector<NumberType>(num_node / world_size));

	std::unordered_map<int, std::vector<NumberType>> count_buffers;
	std::unordered_map<int, std::vector<NumberType>> cooccur_buffers;

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

			// add_count(count, T_nodes[j], myrank, world_size);
			add_count_batch(T_nodes[j], myrank, world_size, count_buffers, local_count);

			for (int l = j; l < T_nodes.size(); l++)
			{
				int a = T_nodes[j];
				int b = T_nodes[l];

				// Increment node pair co-occurences in current set
				// add_occurance(C, T_nodes[j], T_nodes[l], myrank, world_size);
				add_occurance_batch(T_nodes[j], T_nodes[l], myrank, world_size, cooccur_buffers, local_C);
				if (a != b)
					add_occurance_batch(T_nodes[l], T_nodes[j], myrank, world_size, cooccur_buffers, local_C);
				// add_occurance(C, T_nodes[l], T_nodes[j], myrank, world_size);
			}
		}
	}

	// mpi barrier here
	MPI_Barrier(MPI_COMM_WORLD);
	flush_count_messages(count_buffers, myrank, world_size, get_mpi_type<NumberType>());
	flush_occurance_messages(cooccur_buffers, myrank, world_size, get_mpi_type<NumberType>());

	// now receive updates
	receive_count_messages(local_count, myrank, world_size, get_mpi_type<NumberType>());
	receive_occurance_messages(local_C, myrank, world_size, get_mpi_type<NumberType>());

	// // Seed Selection
	// int y_idx = -1;	  // Index of node with highest count
	// int y_count = -1; // Count of most occurring node
	// int y_node = -1;  // Most occurring node
	// for (int i = 0; i < k; i++)
	// {
	// 	// TODO: get global argmax of count vector
	// 	// Get most occurring node
	// 	for (int j = 0; j < num_node; j++)
	// 	{
	// 		if (count[j] > y_count)
	// 		{
	// 			y_count = count[j];
	// 			y_idx = j;
	// 		}
	// 	}
	// 	y_node = R[y_idx];
	// 	count[y_idx] = 0;
	// 	S.push_back(y_node);

	// 	// Update counts for every node i
	// 	for (int i = 0; i < N; i++)
	// 		count[i] = std::max(0, count[i] - local_C[y][i]);
	// }

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
	MPI_Type_free(&mpi2long_type);
	return S;
}
