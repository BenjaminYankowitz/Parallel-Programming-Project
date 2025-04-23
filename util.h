
#ifndef UTIL_H
#define UTIL_H

#include <unordered_map>
#include <vector>
#include "EdgeInfo.h"
typedef EdgeType::IntType NumberType;

#include <mpi.h>

template <typename T>
MPI_Datatype get_mpi_type();

template <>
inline MPI_Datatype get_mpi_type<int>() { return MPI_INT; }

template <>
inline MPI_Datatype get_mpi_type<long>() { return MPI_LONG; }

template <>
inline MPI_Datatype get_mpi_type<unsigned long>() { return MPI_UNSIGNED_LONG; }

// template <>
// MPI_Datatype get_mpi_type<size_t>()
// {
//     static_assert(sizeof(size_t) == sizeof(unsigned long), "Handle size_t mapping carefully!");
//     return MPI_UNSIGNED_LONG;
// }


// example usage
// MPI_Datatype mpi_type = get_mpi_type<NumberType>();

//////////////// define new MPI_DATATYPE//////////////////////
struct MaxLoc
{
    NumberType value;
    NumberType node;
};

inline MPI_Datatype create_MPI_2NUM()
{
    MPI_Datatype type;
    MaxLoc temp;
    int blocklengths[2] = {1, 1};

    // Use your existing helper to deduce MPI type
    MPI_Datatype mpi_num_type = get_mpi_type<NumberType>();
    MPI_Datatype types[2] = {mpi_num_type, mpi_num_type};
    MPI_Aint displacement[2];

    MPI_Aint base;
    MPI_Get_address(&temp, &base);
    MPI_Get_address(&temp.value, &displacement[0]);
    MPI_Get_address(&temp.node, &displacement[1]);

    displacement[0] -= base;
    displacement[1] -= base;

    MPI_Type_create_struct(2, blocklengths, displacement, types, &type);
    MPI_Type_commit(&type);
    return type;
}

inline void maxloc_reduce(void *in, void *inout, int *len, MPI_Datatype *datatype)
{
    MaxLoc *in_vals = static_cast<MaxLoc *>(in);
    MaxLoc *inout_vals = static_cast<MaxLoc *>(inout);
    for (int i = 0; i < *len; ++i)
    {
        if (in_vals[i].value > inout_vals[i].value)
        {
            inout_vals[i] = in_vals[i];
        }
    }
}

NumberType find_global_max_node_id(int local_size,
                                   int world_size,
                                   int rank,
                                   NumberType (*id_local_to_global)(NumberType, int, int))
{
    // 1. Compute local max
    NumberType local_max = 0;
    for (int local_idx = 0; local_idx < local_size; ++local_idx)
    {
        NumberType gid = id_local_to_global(local_idx, world_size, rank);
        local_max = std::max(local_max, gid);
    }

    // 2. Reduce across all ranks
    NumberType global_max;
    MPI_Allreduce(&local_max,
                  &global_max,
                  1,
                  get_mpi_type<NumberType>(),
                  MPI_MAX,
                  MPI_COMM_WORLD);

    return global_max;
}

//////////////// util used in selectSeed //////////////////////

// function to buffer collection of message to be sent in bigger batch
inline void add_count_batch(NumberType node_index,
                            int myrank,
                            int world_size,
                            std::unordered_map<int, std::vector<NumberType>> &count_buffers,
                            std::vector<NumberType> &local_count)
{
    int destination = node_index % world_size;
    if (destination == myrank)
    {
        NumberType local_index = node_index / world_size;
        local_count[local_index]++;
        return;
    }
    count_buffers[destination].push_back(node_index);
}

// function to buffer collection of message to be sent in bigger batch
inline void add_occurance_batch(NumberType row_index, NumberType col_index,
                                int myrank,
                                int world_size,
                                std::unordered_map<int, std::vector<NumberType>> &cooccur_buffers,
                                std::vector<std::vector<NumberType>> &local_C)
{
    int destination = col_index % world_size;
    if (destination == myrank)
    {
        // process locally
        NumberType local_index = col_index / world_size;
        local_C[row_index][local_index]++;
        return;
    }
    cooccur_buffers[destination].push_back(row_index);
    cooccur_buffers[destination].push_back(col_index);
}

inline void flush_count_messages(std::unordered_map<int, std::vector<NumberType>> &count_buffers,
                                 int myrank, int world_size,
                                 MPI_Datatype mpi_type)
{
    for (auto &[dest, buffer] : count_buffers)
    {
        MPI_Send(buffer.data(), buffer.size(), mpi_type, dest, 0, MPI_COMM_WORLD);
    }

    // Send DONE message to each rank ≠ myrank
    NumberType sentinel = static_cast<NumberType>(-1);
    for (int rank = 0; rank < world_size; ++rank)
    {
        if (rank != myrank)
        {
            MPI_Send(&sentinel, 1, mpi_type, rank, 99, MPI_COMM_WORLD); // tag 99 = DONE
        }
    }
}

inline void flush_occurance_messages(std::unordered_map<int, std::vector<NumberType>> &cooccur_buffers,
                                     int myrank, int world_size, MPI_Datatype mpi_type)
{
    for (auto &[dest, buffer] : cooccur_buffers)
    {
        MPI_Send(buffer.data(), buffer.size(), mpi_type, dest, 1, MPI_COMM_WORLD);
    }

    // Send sentinel (-1, -1) to all other ranks
    NumberType sentinel[2] = {static_cast<NumberType>(-1), static_cast<NumberType>(-1)};
    for (int rank = 0; rank < world_size; ++rank)
    {
        if (rank != myrank)
        {
            MPI_Send(sentinel, 2, mpi_type, rank, 98, MPI_COMM_WORLD); // tag 98 = DONE
        }
    }
}

inline void receive_count_messages(std::vector<NumberType> &local_count,
                                   int myrank, int world_size, MPI_Datatype mpi_type)
{
    MPI_Status status;
    int done_count = 0;

    while (done_count < world_size - 1)
    {
        // Wait for any incoming message (either data or DONE)
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        int tag = status.MPI_TAG;
        if (tag == 99)
        {
            // DONE message: exactly one element
            NumberType sentinel;
            MPI_Recv(&sentinel, 1, mpi_type, status.MPI_SOURCE, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            done_count++;
        }
        else
        {
            // Regular count message: determine length dynamically
            int count;
            MPI_Get_count(&status, mpi_type, &count);
            std::vector<NumberType> buf(count);

            MPI_Recv(buf.data(), count, mpi_type, status.MPI_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Apply all received updates
            for (int i = 0; i < count; ++i)
            {
                NumberType node = buf[i];
                int local_idx = node / world_size;
                if (local_idx >= 0 && local_idx < static_cast<int>(local_count.size()))
                {
                    local_count[local_idx]++;
                }
            }
        }
    }
}

inline void receive_occurance_messages(std::vector<std::vector<int>> &local_C,
                                       int myrank, int world_size, MPI_Datatype mpi_type)
{
    MPI_Status status;
    int done_count = 0;

    while (done_count < world_size - 1)
    {
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int tag = status.MPI_TAG;

        if (tag == 98)
        {
            NumberType sentinel[2];
            MPI_Recv(sentinel, 2, mpi_type, status.MPI_SOURCE, 98, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            done_count++;
        }
        else
        {
            int count;
            MPI_Get_count(&status, mpi_type, &count);
            std::vector<NumberType> buf(count);
            MPI_Recv(buf.data(), count, mpi_type, status.MPI_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // buf holds [row0,col0,row1,col1,...]
            for (int i = 0; i < count; i += 2)
            {
                NumberType row = buf[i];
                NumberType col = buf[i + 1];
                int local_col = col / world_size;
                if (row >= 0 && row < local_C.size() &&
                    local_col >= 0 && local_col < local_C[row].size())
                {
                    local_C[row][local_col]++;
                }
            }
        }
    }
}

inline NumberType find_global_argmax(const std::vector<NumberType> &local_count, int myrank, int world_size,
                                     MPI_Datatype maxloc_type, MPI_Op maxloc_op)
{
    MaxLoc local = {-1, -1};

    for (size_t i = 0; i < local_count.size(); ++i)
    {
        if (local_count[i] > local.value)
        {
            local.value = local_count[i];
            local.node = static_cast<NumberType>(i * world_size + myrank);
        }
    }

    MaxLoc global;
    MPI_Allreduce(&local, &global, 1, maxloc_type, maxloc_op, MPI_COMM_WORLD);
    return global.node;
}

#endif


