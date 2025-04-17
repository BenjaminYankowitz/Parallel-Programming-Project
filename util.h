
#ifndef UTIL_H
#define UTIL_H

#include <unordered_map>
#include "EdgeInfo.h"
typedef EdgeType::IntType NumberType;


#include <mpi.h>



template <typename T>
MPI_Datatype get_mpi_type();

template <>
MPI_Datatype get_mpi_type<int>() { return MPI_INT; }

template <>
MPI_Datatype get_mpi_type<long>() { return MPI_LONG; }

template <>
MPI_Datatype get_mpi_type<unsigned long>() { return MPI_UNSIGNED_LONG; }

template <>
MPI_Datatype get_mpi_type<size_t>()
{
    static_assert(sizeof(size_t) == sizeof(unsigned long), "Handle size_t mapping carefully!");
    return MPI_UNSIGNED_LONG;
}



//////////////// util used in selectSeed //////////////////////

// function to buffer collection of message to be sent in bigger batch
void add_count_batch(NumberType node_index,
                     int myrank,
                     int world_size,
                     std::unordered_map<int, std::vector<NumberType>> &count_buffers,
                     std::vector<NumberType> &local_count)
{
    int destination = node_index % world_size;
    if (destination == myrank)
    {
        local_count[node_index]++;
        return; 
    }
    count_buffers[destination].push_back(node_index);
}

// function to buffer collection of message to be sent in bigger batch
void add_occurance_batch(NumberType row_index, NumberType col_index,
                         int myrank,
                         int world_size,
                         std::unordered_map<int, std::vector<NumberType>> &cooccur_buffers,
                         std::vector<std::vector<NumberType>> local_C)
{
    int destination = row_index % world_size;
    if (destination == myrank)
    {
        // process locally
        local_C[row_index][col_index]++;
        return; 
    }
    cooccur_buffers[destination].push_back(row_index);
    cooccur_buffers[destination].push_back(col_index);
}


void flush_count_messages(std::unordered_map<int, std::vector<NumberType>>& count_buffers,
                          MPI_Datatype mpi_type)
{
    for (auto& [dest, buffer] : count_buffers) {
        MPI_Send(buffer.data(), buffer.size(), mpi_type, dest, 0, MPI_COMM_WORLD);
    }
    count_buffers.clear();
}


void flush_occurance_messages(std::unordered_map<int, std::vector<NumberType>>& cooccur_buffers,
                              MPI_Datatype mpi_type)
{
    for (auto& [dest, buffer] : cooccur_buffers) {
        MPI_Send(buffer.data(), buffer.size(), mpi_type, dest, 1, MPI_COMM_WORLD);
    }
    cooccur_buffers.clear();
}

void receive_count_messages(std::vector<NumberType>& local_count, int myrank, MPI_Datatype mpi_type)
{
    MPI_Status status;
    while (true) {
        NumberType buffer[1024];
        int flag;
        MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
        if (!flag) break;

        int count;
        MPI_Get_count(&status, mpi_type, &count);
        MPI_Recv(buffer, count, mpi_type, status.MPI_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = 0; i < count; ++i) {
            local_count[buffer[i]]++;
        }
    }
}

void receive_occurance_messages(std::vector<std::vector<int>>& local_C, int myrank, MPI_Datatype mpi_type)
{
    MPI_Status status;
    while (true) {
        NumberType buffer[1024];
        int flag;
        MPI_Iprobe(MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &flag, &status);
        if (!flag) break;

        int count;
        MPI_Get_count(&status, mpi_type, &count);
        MPI_Recv(buffer, count, mpi_type, status.MPI_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = 0; i < count; i += 2) {
            NumberType a = buffer[i];
            NumberType b = buffer[i + 1];
            local_C[a][b]++;
        }
    }
}












#endif


// example usage
// MPI_Datatype mpi_type = get_mpi_type<NumberType>();