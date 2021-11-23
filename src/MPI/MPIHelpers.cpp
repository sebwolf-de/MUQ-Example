#include "MPIHelpers.h"
#include <mpi.h>

size_t MPI::get_global_rank() {
  int global_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
  return global_rank;
}
