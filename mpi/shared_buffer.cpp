#include <mpi.h>
#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <iostream>

#pragma omp requires unified_shared_memory

int main(int argc, char** argv) {
  MPI_Init(NULL, NULL);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if (world_size != 2) {
    fprintf(stderr, "World size must be 2 for %s\n", argv[0]);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  const int N = 100;
  const int hd = omp_get_initial_device();

  const int num_dev = omp_get_num_devices();

  int* buf = (int *) omp_target_alloc(N*sizeof(int), world_rank % num_dev);
  if (world_rank == 0) {
    #pragma omp target is_device_ptr(buf) device(world_rank % num_dev)
    for (int i=0; i<N; ++i)
         buf[i] = i; 
    MPI_Send(buf, N, MPI_INT, 1, 0, MPI_COMM_WORLD);
  } else if (world_rank == 1) {
    MPI_Recv(buf, N, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i=0; i<N; ++i) 
      assert( buf[i] == i );
  }
  MPI_Finalize();
}

