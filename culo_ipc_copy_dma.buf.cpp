#include <libgen.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cuda.h>

static const char * cudaErrorName(CUresult rc) {
  const char *result;
  if (cuGetErrorName(rc, &result) != CUDA_SUCCESS)
     return "unknown error";
  return result;
}


