#include <iostream>
#include <libgen.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cuda.h>

#define CHILDPROCESSES 1
int sv[CHILDPROCESSES][2];

size_t allocSize = 4096;


#define Elog(fmt,...)                            \
    do {                                         \
         fprintf(stderr, "%s:%d: " fmt "\n",     \
         __FUNCTION__, __LINE__, ##__VA_ARGS__); \
         exit(1);                                \
    } while(0)


static int sendmsg(int pipefd, CUipcMemHandle ipc_mhandle) {
    if (write(pipefd, &ipc_mhandle, sizeof(ipc_mhandle)) != sizeof(ipc_mhandle))
	Elog("failed on write(&ipc_mhandle): %m");
  
    return true;
}

static int reivmsg(int pipefd, CUipcMemHandle *ipc_mhandle) {
     if (read(pipefd, ipc_mhandle, sizeof(*ipc_mhandle)) != sizeof(*ipc_mhandle))
        Elog("failed on read(&ipc_mhandle): %m");

    return true;
}

static const char * cudaErrorName(CUresult rc) {
    const char *result;
    if (cuGetErrorName(rc, &result) != CUDA_SUCCESS)
      return "unknown error";
    return result;
}

inline void initializeProcess(CUdeviceptr  &cuda_devptr){

     CUresult       rc;
     CUdevice       cuda_device;
     CUcontext      cuda_context;
     CUipcMemHandle ipc_mhandle;

     /* init CUDA context */
     rc = cuInit(0);
     if (rc != CUDA_SUCCESS)
          Elog("failed on cuInit: %s", cudaErrorName(rc));
     rc = cuDeviceGet(&cuda_device, 0);
     if (rc != CUDA_SUCCESS)
          Elog("failed on cuDeviceGet: %s", cudaErrorName(rc));

     rc = cuCtxCreate(&cuda_context, 0, cuda_device);
     if (rc != CUDA_SUCCESS)
        Elog("failed on cuCtxCreate: %s", cudaErrorName(rc));
}

void run_client(int commPipe, uint32_t clientId) {
     std::cout << "Client " << clientId << ", process ID: " << std::dec << getpid() << "\n";

     CUresult       rc;
     CUdeviceptr    cuda_devptr;
     CUipcMemHandle ipc_mhandle;

    initializeProcess(cuda_devptr);

    char *heapBuffer = new char[allocSize];
    for (size_t i = 0; i < allocSize; ++i) {
        heapBuffer[i] = static_cast<char>(i + 1);
    }


     close(sv[clientId][1]);
     reivmsg(commPipe, &ipc_mhandle);
     close(sv[clientId][0]);

     rc = cuIpcOpenMemHandle(&cuda_devptr, ipc_mhandle,
                             CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
     if (rc != CUDA_SUCCESS)
        Elog("Failed on cuIpcOpenMemHandle: %s", cudaErrorName(rc));

     rc = cuMemcpyHtoD(cuda_devptr, heapBuffer, allocSize); 
     if (rc != CUDA_SUCCESS)
        Elog("Failed on cuMemcpyHtoD: %s", cudaErrorName(rc));

     rc = cuIpcCloseMemHandle(cuda_devptr);
     if (rc != CUDA_SUCCESS)
	Elog("Failed on cuIpcCloseMemHandle: %s", cudaErrorName(rc));
}

void run_server(bool &validRet) {
     std::cout << "Server process ID " << std::dec << getpid() << "\n";

     CUresult       rc;
     CUdeviceptr    cuda_devptr;
     CUipcMemHandle ipc_mhandle;

    initializeProcess(cuda_devptr);

     /* allocation and export */
     rc = cuMemAlloc(&cuda_devptr, allocSize);
     if (rc != CUDA_SUCCESS)
        Elog("failed on cuMemAlloc: %s", cudaErrorName(rc));

     for (uint32_t i = 0; i < CHILDPROCESSES; i++) {
         // Initialize the IPC buffer
         int value = 3;
         rc =cuMemsetD32(cuda_devptr, value, allocSize/sizeof(value));
         if (rc != CUDA_SUCCESS)
            Elog("failed on cuMemsetD32: %s", cudaErrorName(rc));

         rc = cuIpcGetMemHandle(&ipc_mhandle, cuda_devptr);
         if (rc != CUDA_SUCCESS)
            Elog("failed on cuIpcGetMemHandle: %s", cudaErrorName(rc));

    	 close(sv[i][0]);
         sendmsg(sv[i][1], ipc_mhandle);
         close(sv[i][1]);

        char *heapBuffer = new char[allocSize];
        for (size_t i = 0; i < allocSize; ++i) {
            heapBuffer[i] = static_cast<char>(i + 1);
        }


         // Wait for child to exit
         int child_status;
         pid_t clientPId = wait(&child_status);
         if (clientPId <= 0) {
            std::cerr << "Client terminated abruptly with error code " << strerror(errno) << "\n";
            std::terminate();
         }

        void *validateBuffer = new char[allocSize];;
        rc = cuMemcpyDtoH (validateBuffer, cuda_devptr, allocSize); 
        if (rc != CUDA_SUCCESS)
        	Elog("failed on cuMemcpyDtoH: %s", cudaErrorName(rc));

        validRet = (0 == memcmp(heapBuffer, validateBuffer, allocSize));
     }
}

int main(int argc, char *argv[]) {

    bool outputValidationSuccessful;

    pid_t childPids[CHILDPROCESSES];
    for (uint32_t i = 0; i < CHILDPROCESSES; i++) {
        if (pipe(sv[i])!=0) {       
           perror("pipe");
           exit(1);
       }

        childPids[i] = fork();
        if (childPids[i] < 0) {
            perror("fork");
            exit(1);
        } else if (childPids[i] == 0) {
            run_client(sv[i][0], i);
            exit(0);
        }
    }
    run_server(outputValidationSuccessful);

    std::cout << "\nZello IPC Results validation "
              << (outputValidationSuccessful ? "PASSED" : "FAILED")
              << std::endl;

}
