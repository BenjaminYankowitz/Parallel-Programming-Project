#include <mpi.h>
#include <vector>
#include <iostream>
#include "util.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Datatype maxloc_type = create_MPI_2NUM();
    MPI_Op maxloc_op;
    MPI_Op_create(&maxloc_reduce, /*commute=*/1, &maxloc_op);

    std::vector<NumberType> local_count(3, 0);
    for (int i = 0; i < 3; ++i)
    {
        NumberType global_id = i * size + rank;
        local_count[i] = (global_id == 5) ? 99 : global_id;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    for (int r = 0; r < size; ++r)
    {
        if (r == rank)
        {
            std::cout << "Rank " << rank << " local nodes and counts:\n";
            for (int i = 0; i < local_count.size(); ++i)
            {
                NumberType global_node_id = i * size + rank;
                std::cout << "  Node " << global_node_id << " => Count: " << local_count[i] << "\n";
            }
            std::cout << std::flush;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    NumberType max_node = find_global_argmax(local_count, rank, size, maxloc_type, maxloc_op);
    if (rank == 0)
    {
        std::cout << "Global max at node: " << max_node << std::endl;
    }

    MPI_Op_free(&maxloc_op);
    MPI_Type_free(&maxloc_type);
    MPI_Finalize();
    return 0;
}
