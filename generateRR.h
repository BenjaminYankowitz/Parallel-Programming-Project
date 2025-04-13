#include <mpi.h>
#include <set>
#include <list>
#include <queue>
#include <vector>
#include <algorithm>
#include <random>
#include "rand_gen.h"
#include <iostream>

// graph collect node,edge information in terms of adjacency matrix
// adj_vector[i] is a vector of neighbors of node i
class graph
{
public:
    std::vector<std::vector<long> > adj_vector;

    size_t size()
    {
        return adj_vector.size();
    }
};

struct frontier_tuple
{
    long node_id;
    long walk_id;
    long level;
};

// convert local node id to global node id
long id_local_to_global(long node_id, int world_size, int myrank)
{
    return node_id * world_size + myrank;
}

// convert global node id to local node id
long id_global_to_local(long node_id, int world_size, int myrank)
{
    return (node_id - myrank) / world_size;
}

void MPI_send_frontier_raw(long node_id, long walk_id, long level, int dest, int tag)
{
    long buffer[3] = {node_id, walk_id, level};
    MPI_Send(buffer, 3, MPI_LONG, dest, tag, MPI_COMM_WORLD);
}

inline bool operator<(const frontier_tuple &lhs, const frontier_tuple &rhs)
{
    if (lhs.node_id != rhs.node_id)
        return lhs.node_id < rhs.node_id;
    if (lhs.walk_id != rhs.walk_id)
        return lhs.walk_id < rhs.walk_id;
    return lhs.level < rhs.level;
}

void setup_generator(int myrank)
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
void select_random_nodes(const std::vector<int> &nodes, unsigned long num_sample, std::queue<frontier_tuple> &frontier)
{
    if (nodes.empty() || num_sample <= 0)
    {
        return;
    }
    unsigned long max_length = nodes.size();

    for (long i = 0; i < num_sample; i++)
    {
        // randomly pick the vector index n times
        // random from [0, max_length-1]
        uint32_t x = genrand_int_n(max_length - 1);

        frontier_tuple new_tuple = {nodes[x], i, 0};
        frontier.push(new_tuple);
    }
}

// max_size = number of nodes in subgraph
void select_random_nodes(unsigned long max_size, unsigned long num_sample, std::queue<frontier_tuple> &frontier, int myrank, int world_size)
{
    for (long i = 0; i < num_sample; i++)
    {
        // randomly pick the vector index n times
        // random from [0, max_length-1]
        uint32_t x = genrand_int_n(max_size - 1);

        // printf("got x value = %d", x);

        frontier_tuple new_tuple = {x * world_size + myrank, i, 0};
        frontier.push(new_tuple);
    }
}

std::vector<long> neighbor_of(graph mygraph, long node, int world_size, int myrank)
{
    // TODO: implement getting neighbor
    // std::vector<long> neighbor;
    // neighbor.push_back((node + 1) % (mygraph.size() * world_size));
    // neighbor.push_back((node + 2) % (mygraph.size() * world_size));

    return mygraph.adj_vector[id_global_to_local(node, world_size, myrank)];
}

std::vector<std::set<long>> generate_RR(graph sub_graph, unsigned long num_sample, int myrank, int world_size)
{
    unsigned long num_node = sub_graph.size();
    // error check
    if (num_node == 0)
    {
        // nothing to do since subgraph is eempty
        std::vector<std::set<long>> RR(num_node);
        return RR;
    }

    std::cout << "Rank [" << myrank << "]" << "setting up RNG\n";
    setup_generator(myrank);

    std::vector<std::set<long>> RR(num_node);
    std::queue<frontier_tuple> frontier;
    std::queue<frontier_tuple> next_frontier;

    // randomize <num_sample> starting position
    std::cout << "Rank [" << myrank << "]" << "selecting starting points\n";
    // select_random_nodes(sub_graph.nodes, num_sample, frontier);
    select_random_nodes(num_node, num_sample, frontier, myrank, world_size);

    long current_level = 0;
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

            std::cout << "Rank [" << myrank << "] level: " << current_level << " (" << tuple.node_id << "," << tuple.walk_id << "," << tuple.level << ")\n";

            long index = (long)((tuple.node_id - myrank) / world_size);
            RR[index].insert(tuple.walk_id);

            double cutoff = genrand_real1(); // random 0 to 1
            std::vector<long> neighbors = neighbor_of(sub_graph, tuple.node_id, world_size, myrank);
            std::cout << "Rank [" << myrank << "]" << "getting neighbor of " << tuple.node_id << "\n";
            for (long &neigh_node_id : neighbors)
            {
                std::cout << "[" << myrank << "]" << " neighbor = " << neigh_node_id << " cutoff = " << cutoff << "\n";

                // TODO: get the edge weight between 2 node
                // if cutoff < edge(neigh_node_id, tuple.node_id)
                if (cutoff < 0.5)
                {
                    more_work_to_do = 1;    // notifying others
                    all_finish = 0;         //notifying self
                    
                    int destination = neigh_node_id % world_size;
                    if (destination == myrank)
                    {
                        // this neighbor node is in my subgraph
                        std::cout << "push to self\n";
                        frontier_tuple new_tuple = {neigh_node_id, tuple.walk_id, tuple.level + 1};
                        next_frontier.push(new_tuple);
                    }
                    else
                    {
                        // TODO
                        // send to other process
                        
                        std::cout << "push to remote [" << destination << "]\n";
                        MPI_send_frontier_raw(neigh_node_id, tuple.walk_id, tuple.level + 1, destination, 0);
                    }
                    break;
                }
            }

            std::cout << "Rank [" << myrank << "] DONE level: " << current_level << "\n";
        }

        MPI_Barrier(MPI_COMM_WORLD);
        // local work is finish, notify the others

        // first notify other rank that it is finish sending
        long stop[3] = {-2 + more_work_to_do, 0, 0};
        // node_id == -1 as special signal => "I'm done. let's move to next level"
        // node_id == -2 as special signal => "I'm done. let's terminate"
        for (int rank = 0; rank < world_size; rank++)
        {
            if (rank != myrank)
            {
                MPI_Send(stop, 3, MPI_LONG, rank, 0, MPI_COMM_WORLD);
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
                long buffer[3];
                MPI_Recv(buffer, 3, MPI_LONG, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, &status);

                if (buffer[0] < 0)
                {
                    if (buffer[0] == -1)
                    {
                        // someone is not done yet
                        all_finish = 0;
                    }
                    finished_senders++;
                    printf("[DEBUG] Receiver: sender %d is done [%ld]\n", status.MPI_SOURCE, buffer[0]);
                }
                else
                {
                    frontier_tuple data = {buffer[0], buffer[1], buffer[2]};
                    next_frontier.push(data);
                    printf("[%d] Receiver got from %d: id=%ld, num=%ld, level=%ld\n",
                           myrank, status.MPI_SOURCE, data.node_id, data.walk_id, data.level);
                }
            }
        }
        // MPI_Barrier(MPI_COMM_WORLD);

        // swap the queue
        std::queue<frontier_tuple> temp = frontier;
        frontier = next_frontier;
        next_frontier = temp; // next_frontier should now be empty
        if (all_finish == 0)
        {

            std::cout << "Rank [" << myrank << "] current_lvl " << current_level << " ready to move to level: " << current_level + 1 << "\n";
        }
        else
        {
            std::cout << "Rank [" << myrank << "] current_lvl " << current_level << " ready to terminate" << "\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
        current_level++;
    }

    std::cout << "Rank [" << myrank << "] has terminated at " << current_level - 1 << "\n";
    return RR;
}
