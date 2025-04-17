
#ifndef UTIL_H
#define UTIL_H

#include <mpi.h>

template<typename T>
MPI_Datatype get_mpi_type();

template<>
MPI_Datatype get_mpi_type<int>() { return MPI_INT; }

template<>
MPI_Datatype get_mpi_type<long>() { return MPI_LONG; }

template<>
MPI_Datatype get_mpi_type<unsigned long>() { return MPI_UNSIGNED_LONG; }

template<>
MPI_Datatype get_mpi_type<size_t>() {
    static_assert(sizeof(size_t) == sizeof(unsigned long), "Handle size_t mapping carefully!");
    return MPI_UNSIGNED_LONG;
}

#endif

// example usage
// MPI_Datatype mpi_type = get_mpi_type<NumberType>();