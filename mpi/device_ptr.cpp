#include <mpi.h>
#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <iostream>

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

  int* dev = (int *) omp_target_alloc_device(N*sizeof(int), world_rank % num_dev);
  int * buf = (int*) malloc(N*sizeof(int));

  if (world_rank == 0) {
    for (int i=0; i<N; ++i)
         buf[i] = i; 
    omp_target_memcpy(dev, buf, N*sizeof(int), 0,0,  world_rank % num_dev, hd); 
    MPI_Send(&dev, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
  } else if (world_rank == 1) {
    MPI_Recv(&dev, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    omp_target_memcpy(buf,dev, N*sizeof(int), 0,0,  hd, world_rank % num_dev);
    for (int i=0; i<N; ++i) 
      assert( buf[i] == i );
  }
  MPI_Finalize();
}

