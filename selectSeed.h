#include <mpi.h> 
#include <vector> 
#include <set> 
#include <algorithm> 
#include <iostream>


std::vector<int> selectSeed2D(std::vector<std::set<long>> R, int k, int N, int myrank, int world_size) 
{ 
	std::vector<int> S; // Output set
    S.reserve(k);
	std::vector<int> count(N, 0); // Count array
	std::vector<std::vector<int>> C(N, std::vector<int>(N, 0)); // Co-occurrence matrix
	 
	// Preprocessing step
	for (int i = 0; i < N; i++) 
	{
		// A single RRR set
		std::set<long> T = R[i];
		
	    // Build matrix 
		std::vector<int> T_nodes(T.begin(), T.end());
		for (int j = 0; j < T_nodes.size(); j++) 
		{
			// Count nodes in current RRR set
			if (node < N) 
			    count[node] += 1; // serial for now - change later
			
			for (int l = j; l < T_nodes.size(); l++) 
			{
			    int a = T_nodes[j];
			    int b = T_nodes[l];
			    if (a < N && b < N) 
			    {
			    	// Increment node pair co-occurences in current set
			        C[a][b] += 1; // serial for now - change later
			        if (a != b)
			            C[b][a] += 1; 
			    }
			}
		}
	}

	// mpi barrier here

	// Seed Selection 
	int y_idx = -1; // Index of node with highest count
    int y_count = -1; // Count of most occurring node
    int y_node = -1; // Most occurring node 
	for (int i = 0; i < k; i++) 
	{
		// Get most occurring node
        for (int j = 0; j < N; j++) 
        {
            if (count[j] > y_count) 
            {
                y_count = count[j];
                y_idx = j;
            }
        }
        y_node = R[y_idx];
        count[y_idx] = 0;
        S.push_back(y_node);

        // Update counts for every node i
        for (int i = 0; i < N; i++) 
            count[i] = std::max(0, count[i] - C[y][i]);
    }
	

	return S;
}

