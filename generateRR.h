#ifndef GENERATERR_H
#define GENERATERR_H

#include <mpi.h>
#include <algorithm>
#include <iostream>
#include <list>
#include <queue>
#include <random>
#include <set>
#include <unordered_set>
#include <vector>
#include "rand_gen.h"
#include "EdgeInfo.h"

// typedef long NumberType;
typedef EdgeType::IntType NumberType;

// graph collect node,edge information in terms of adjacency matrix
// adj_vector[i] is a vector of neighbors of node i
class graph
{
public:
    std::vector<std::vector<EdgeType>> adj_vector;

    size_t size()
    {
        return adj_vector.size();
    }
};

struct frontier_tuple
{
    NumberType node_id;
    NumberType walk_id;
    NumberType level;
};

// convert local node id to global node id
inline NumberType id_local_to_global(NumberType node_id, int world_size, int myrank)
{
    return node_id * world_size + myrank;
}

// convert global node id to local node id
inline NumberType id_global_to_local(NumberType node_id, int world_size, int myrank)
{
    return (node_id - myrank) / world_size;
}

inline void MPI_send_frontier_raw(NumberType node_id, NumberType walk_id, NumberType level, int dest, int tag)
{
    NumberType buffer[3] = {node_id, walk_id, level};
    static_assert(std::is_same<NumberType, long>::value || std::is_same<NumberType, int>::value);
    if (std::is_same<NumberType, long>::value)
    {
        // using num type = long
        MPI_Send(buffer, 3, MPI_LONG, dest, tag, MPI_COMM_WORLD);
    }
    else if (std::is_same<NumberType, int>::value)
    {
        // using num type = int
        MPI_Send(buffer, 3, MPI_INT, dest, tag, MPI_COMM_WORLD);
    }
    else
    {
        std::cout << "NumberType is not a long or int\n";
        exit(-1);
    }
}

inline bool operator<(const frontier_tuple &lhs, const frontier_tuple &rhs)
{
    if (lhs.node_id != rhs.node_id)
        return lhs.node_id < rhs.node_id;
    if (lhs.walk_id != rhs.walk_id)
        return lhs.walk_id < rhs.walk_id;
    return lhs.level < rhs.level;
}

inline void setup_generator(int myrank)
{
    const unsigned long length = 624;
    unsigned long g_init[length];

    for (int i = 0; i < length; i++)
    {
        g_init[i] = ((unsigned long)(1 << 31) - 1) - (((unsigned long)(1 << 21) - 1) * i) - (((unsigned long)(1 << 15) - 1) * i) - (1023 * myrank); // add MPI rank adjustment
    }
    init_by_array(g_init, length);
}

// randomly select n nodes from vector and put in into frontier
// this will randomly select WITH REPLACEMENT (possible duplicate)
inline void select_random_nodes(const std::vector<int> &nodes, unsigned long num_sample, std::queue<frontier_tuple> &frontier)
{
    if (nodes.empty() || num_sample <= 0)
    {
        return;
    }
    unsigned long max_length = nodes.size();

    for (NumberType i = 0; i < num_sample; i++)
    {
        // randomly pick the vector index n times
        // random from [0, max_length-1]
        uint32_t x = genrand_int_n(max_length - 1);

        frontier_tuple new_tuple = {nodes[x], i, 0};
        frontier.push(new_tuple);
    }
}

// max_size = number of nodes in subgraph
inline void select_random_nodes(unsigned long max_size, unsigned long num_sample, std::queue<frontier_tuple> &frontier, int myrank, int world_size)
{

    for (NumberType i = 0; i < num_sample; i++)
    {
        // randomly pick the vector index n times
        // random from [0, max_length-1]
        uint32_t x = genrand_int_n(max_size - 1);

        // printf("got x value = %d", x);

        frontier_tuple new_tuple = {static_cast<NumberType>(x * world_size + myrank), i, 0};
        frontier.push(new_tuple);
    }
}

inline std::vector<EdgeType> neighbor_of(graph mygraph, NumberType node, int world_size, int myrank)
{
    return mygraph.adj_vector[id_global_to_local(node, world_size, myrank)];
}

inline std::vector<std::set<NumberType>> generate_RR(graph sub_graph, unsigned long num_sample, int myrank, int world_size, bool DEBUG_MODE)
{
    unsigned long num_node = sub_graph.size();
    // error check
    if (num_node == 0)
    {
        // nothing to do since subgraph is eempty
        std::vector<std::set<NumberType>> RR(num_node);
        return RR;
    }

    if (DEBUG_MODE)
    {

        std::cout << "Rank [" << myrank << "]" << "setting up RNG\n";
    }
    setup_generator(myrank);

    std::vector<std::set<NumberType>> RR(num_node);
    std::queue<frontier_tuple> frontier;
    std::queue<frontier_tuple> next_frontier;

    // randomize <num_sample> starting position
    if (DEBUG_MODE)
    {

        std::cout << "Rank [" << myrank << "]" << "selecting starting points\n";
    }
    // select_random_nodes(sub_graph.nodes, num_sample, frontier);
    select_random_nodes(num_node, num_sample, frontier, myrank, world_size);

    NumberType current_level = 0;
    int all_finish = 0;
    while (all_finish == 0)
    {
        // not all processing has finished
        all_finish = 1;
        int more_work_to_do = 0;
        while (!frontier.empty())
        {
            // have local work to do in this level
            frontier_tuple tuple = frontier.front();
            frontier.pop();

            if (DEBUG_MODE)
            {

                std::cout << "Rank [" << myrank << "] level: " << current_level << " (" << tuple.node_id << "," << tuple.walk_id << "," << tuple.level << ")\n";
            }

            NumberType index = (NumberType)((tuple.node_id - myrank) / world_size);
            RR[index].insert(tuple.walk_id);

            double cutoff = genrand_real1(); // random 0 to 1
            std::vector<EdgeType> neighbors = neighbor_of(sub_graph, tuple.node_id, world_size, myrank);

            if (DEBUG_MODE)
            {
                std::cout << "Rank [" << myrank << "]" << "getting neighbor of " << tuple.node_id << "\n";
            }

            for (const EdgeType &neigh_node : neighbors)
            {
                const EdgeType::IntType neigh_node_id = neigh_node.to;
                if (DEBUG_MODE)
                {

                    std::cout << "[" << myrank << "]" << " neighbor = " << neigh_node_id << " cutoff = " << cutoff << "\n";
                }

                // TODO: get the edge weight between 2 node
                // if cutoff < edge(neigh_node_id, tuple.node_id)
                if (cutoff < 0.5)
                {
                    more_work_to_do = 1; // notifying others
                    all_finish = 0;      // notifying self

                    int destination = neigh_node_id % world_size;
                    if (destination == myrank)
                    {
                        // this neighbor node is in my subgraph
                        if (DEBUG_MODE)
                        {

                            std::cout << "push to self\n";
                        }
                        frontier_tuple new_tuple = {neigh_node_id, tuple.walk_id, tuple.level + 1};
                        next_frontier.push(new_tuple);
                    }
                    else
                    {
                        // send to other process
                        if (DEBUG_MODE)
                        {
                            std::cout << "push to remote [" << destination << "]\n";
                        }
                        MPI_send_frontier_raw(neigh_node_id, tuple.walk_id, tuple.level + 1, destination, 0);
                    }
                    break;
                }
            }

            if (DEBUG_MODE)
            {
                std::cout << "Rank [" << myrank << "] DONE level: " << current_level << "\n";
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        // local work is finish, notify the others

        // first notify other rank that it is finish sending
        NumberType stop[3] = {-2 + more_work_to_do, 0, 0};
        // node_id == -1 as special signal => "I'm done. let's move to next level"
        // node_id == -2 as special signal => "I'm done. let's terminate"
        for (int rank = 0; rank < world_size; rank++)
        {
            if (rank != myrank)
            {
                if constexpr (std::is_same<NumberType, long>::value)
                {
                    // using num type = long
                    MPI_Send(stop, 3, MPI_LONG, rank, 0, MPI_COMM_WORLD);
                }
                else if constexpr (std::is_same<NumberType, int>::value)
                {
                    // using num type = int
                    MPI_Send(stop, 3, MPI_INT, rank, 0, MPI_COMM_WORLD);
                }
            }
        }

        // will now start receiving message
        int finished_senders = 0;
        int expected_senders = world_size - 1;
        MPI_Status status;

        // keep waiting for message until we heard from everyone
        while (finished_senders < expected_senders)
        {
            int flag;
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
            if (flag)
            {
                NumberType buffer[3];
                if constexpr (std::is_same<NumberType, long>::value)
                {
                    // using num type = long
                    MPI_Recv(buffer, 3, MPI_LONG, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, &status);
                }
                else if constexpr (std::is_same<NumberType, int>::value)
                {
                    // using num type = int
                    MPI_Recv(buffer, 3, MPI_INT, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, &status);
                }

                if (buffer[0] < 0)
                {
                    if (buffer[0] == -1)
                    {
                        // someone is not done yet
                        all_finish = 0;
                    }
                    finished_senders++;

                    if (DEBUG_MODE)
                    {
                        printf("[DEBUG] Receiver: sender %d is done [%d]\n", status.MPI_SOURCE, buffer[0]);
                    }
                }
                else
                {
                    frontier_tuple data = {buffer[0], buffer[1], buffer[2]};
                    next_frontier.push(data);
                    if (DEBUG_MODE)
                    {
                        printf("[%d] Receiver got from %d: id=%d, num=%d, level=%d\n",
                               myrank, status.MPI_SOURCE, data.node_id, data.walk_id, data.level);
                    }
                }
            }
        }
        // MPI_Barrier(MPI_COMM_WORLD);

        // swap the queue
        std::queue<frontier_tuple> temp = frontier;
        frontier = next_frontier;
        next_frontier = temp; // next_frontier should now be empty
        if (DEBUG_MODE)
        {

            if (all_finish == 0)
            {

                std::cout << "Rank [" << myrank << "] current_lvl " << current_level << " ready to move to level: " << current_level + 1 << "\n";
            }
            else
            {
                std::cout << "Rank [" << myrank << "] current_lvl " << current_level << " ready to terminate" << "\n";
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        current_level++;
    }

    if (DEBUG_MODE)
    {

        std::cout << "Rank [" << myrank << "] has terminated at " << current_level - 1 << "\n";
    }
    return RR;
}

inline std::vector<std::set<NumberType>> invertNodeWalks(const std::vector<std::set<NumberType>> &implicitRRset, NumberType num_total_sample)
{
    std::vector<std::set<NumberType>> walk_nodes(num_total_sample); // each element i-th is set of nodes walk_id i-th has visited

    // Step 3: Fill walk_nodes
    for (int node = 0; node < implicitRRset.size(); ++node)
    {
        for (NumberType walk_id : implicitRRset[node])
        {
            std::cout << "walk_id " << walk_id << "\n";

            walk_nodes[walk_id].insert(node);
        }
    }

    return walk_nodes;
}

// warning: assume data type as INT
inline std::vector<std::unordered_set<int>> allrank_combineRR(const std::vector<std::set<NumberType>> &local_explicitRR, int myrank, int world_size, NumberType& num_node)
{
    // Convert local unordered_set data to a flat int vector
    std::vector<NumberType> send_data;
    for (NumberType walk_id = 0; walk_id < local_explicitRR.size(); ++walk_id)
    {
        for (NumberType node : local_explicitRR[walk_id])
        {
            // convert local walk_id to global one 
            NumberType walk_id_offset = myrank * world_size;
            send_data.push_back(walk_id + walk_id_offset);
            send_data.push_back(node);
        }
    }

    int local_size = send_data.size();

    // Gather sizes
    std::vector<int> recv_sizes;
    if (myrank == 0)
        recv_sizes.resize(world_size);
    MPI_Gather(&local_size, 1, MPI_INT, recv_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Gather data
    std::vector<int> displs, recv_data;
    if (myrank == 0)
    {
        displs.resize(world_size);
        int total_size = 0;
        for (int i = 0; i < world_size; ++i)
        {
            displs[i] = total_size;
            total_size += recv_sizes[i];
        }
        recv_data.resize(total_size);
    }

    MPI_Gatherv(send_data.data(), local_size, MPI_INT,
                recv_data.data(), recv_sizes.data(), displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);

    // Reconstruct explicitRR_global (on rank 0)
    std::vector<std::unordered_set<int>> explicitRR_global;
    std::unordered_set<int> unique_nodes;
    if (myrank == 0)
    {
        for (size_t i = 0; i < recv_data.size(); i += 2)
        {
            int walk_id = recv_data[i];
            int node = recv_data[i + 1];
            if (walk_id >= explicitRR_global.size())
            {
                explicitRR_global.resize(walk_id + 1);
            }
            explicitRR_global[walk_id].insert(node);

            unique_nodes.insert(node);
        }
        num_node = unique_nodes.size();
    }

    MPI_Bcast(&num_node, 1, MPI_INT, 0, MPI_COMM_WORLD);



    return explicitRR_global;
}

inline void distribute_walks_cyclic(
    const std::vector<std::unordered_set<int>> *combined_RR, // Only non-null on rank 0
    std::vector<std::unordered_set<int>> &explicitRR_distributed,
    int myrank,
    int world_size)
{

    if (myrank == 0)
    {
        // Build send buffers for each rank
        std::vector<std::vector<int>> send_buffers(world_size);

        for (int walk_id = 0; walk_id < combined_RR->size(); ++walk_id)
        {
            int target_rank = walk_id % world_size;
            send_buffers[target_rank].push_back(walk_id);
            for (int node : (*combined_RR)[walk_id])
            {
                send_buffers[target_rank].push_back(node);
            }
            send_buffers[target_rank].push_back(-1); // sentinel, a value that marks the end of a set
        }

        for (int dest_rank = 1; dest_rank < world_size; ++dest_rank)
        {
            int size = static_cast<int>(send_buffers[dest_rank].size());
            // send the "size" of data we are about to send
            MPI_Send(&size, 1, MPI_INT, dest_rank, 0, MPI_COMM_WORLD); 
            // actually sending the data
            MPI_Send(send_buffers[dest_rank].data(), size, MPI_INT, dest_rank, 1, MPI_COMM_WORLD);
        }

        // Rank 0 unpacks its own chunk
        const auto &buf = send_buffers[0];
        int i = 0;
        while (i < buf.size())
        {
            int walk_id = buf[i++];
            std::unordered_set<int> nodes;
            while (i < buf.size() && buf[i] != -1)
            {
                nodes.insert(buf[i++]);
            }
            ++i; // skip sentinel
            if (walk_id >= explicitRR_distributed.size())
                explicitRR_distributed.resize(walk_id + 1);
            explicitRR_distributed[walk_id] = std::move(nodes);
        }
    }
    else
    {
        // receive how many data will be sent to me
        int recv_size;
        MPI_Status status;
        MPI_Recv(&recv_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

        // actually receiving data
        std::vector<int> recv_buffer(recv_size);
        MPI_Recv(recv_buffer.data(), recv_size, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);

        int i = 0;
        while (i < recv_buffer.size())
        {
            int walk_id = recv_buffer[i++];
            std::unordered_set<int> nodes;
            while (i < recv_buffer.size() && recv_buffer[i] != -1)
            {
                nodes.insert(recv_buffer[i++]);
            }
            ++i; // skip sentinel, a special mark to say current set ends here
            if (walk_id >= explicitRR_distributed.size())
                explicitRR_distributed.resize(walk_id + 1);
            explicitRR_distributed[walk_id] = std::move(nodes);
        }
    }
}

#endif