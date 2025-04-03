// #include <mpi.h>
#include <set>
#include <list>
#include <queue>
#include <vector>
#include <algorithm>
#include <random>

class graph
{
public:
    std::vector<int> nodes;

};

struct frontier_tuple
{
    int node_id;
    int walk_id;
    int level;
};

inline bool operator<(const frontier_tuple &lhs, const frontier_tuple &rhs)
{
    if (lhs.node_id != rhs.node_id)
        return lhs.node_id < rhs.node_id;
    if (lhs.walk_id != rhs.walk_id)
        return lhs.walk_id < rhs.walk_id;
    return lhs.level < rhs.level;
}

// randomly select n nodes from vector and put in into frontier
void select_random_nodes(const std::vector<int> &nodes, int n, std::queue<frontier_tuple> &frontier)
{

    if (nodes.empty() || n <= 0)
    {
        return;
    }

    std::vector<int> shuffledNodes = nodes; // Copy to avoid modifying original
    std::random_device rd;
    std::mt19937 g(rd());

    // shuffle the vector
    std::shuffle(shuffledNodes.begin(), shuffledNodes.end(), g);

    for (int i = 0; i < std::min(n, (int)shuffledNodes.size()); ++i)
    {
        frontier.push({shuffledNodes[i], i, 0}); // (node_id, walk_id, level)
    }
}

std::vector<int> neighbor_of(graph mygraph, int node)
{
    // TODO: implement getting neighbor
    std::vector<int> neighbor;
    return neighbor;
}



std::vector<std::set<int>> generate_RR(graph sub_graph, int num_sample, int myrank, int world_size)
{
    int num_node = sub_graph.nodes.size();
    std::vector<std::set<int>> RR(num_node);
    std::queue<frontier_tuple> frontier;

    // randomize <num_sample> starting position
    // select_random_nodes(sub_graph.nodes, num_sample, frontier);
    int current_level = 0;
    int latest_level = 0;
    while (!frontier.empty())
    {
        frontier_tuple tuple = frontier.front();
        latest_level = tuple.level;
        if (latest_level != current_level){
            // TODO: call a barier here
            current_level++;
        }
        frontier.pop();

        RR[tuple.node_id].insert(tuple.walk_id);

        double cutoff = (double)rand() / RAND_MAX; // random 0 to 1
        std::vector<int> neighbors = neighbor_of(sub_graph, tuple.node_id);
        for (int &neigh_node_id : neighbors)
        {
            // TODO: get the edge weight between 2 node
            // if cutoff < edge(neigh_node_id, tuple.node_id)
            if (cutoff < 0.5)
            {
                if (neigh_node_id % world_size == myrank) {
                    // this neighbor node is in my subgraph
                    frontier.push({tuple.node_id, tuple.walk_id, tuple.level + 1});
                }
                else {
                    // TODO
                    // send to other process
                }
                break;
            }
        }

        // TODO: need to wait when it is the last one in current level
        if (frontier.empty()) {
            // need a case to wait until all frontier in every rank is empty
        }
    }
}
