#include "zello_common.h"

#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "omp.h"


#include <CL/sycl.hpp>
// One should read the manual
// https://github.com/smaslov-intel/llvm/blob/master/sycl/doc/extensions/LevelZeroBackend/LevelZeroBackend.md
#include "level_zero/ze_api.h"
#include <CL/sycl/backend/level_zero.hpp>

#define CHILDPROCESSES 1

int sv[CHILDPROCESSES][2];
extern bool verbose;
bool verbose = false;

size_t allocSize = 4096 + 7; // +7 to break alignment and make it harder

static int sendmsg_fd(int socket, int fd) {
    char sendBuf[sizeof(ze_ipc_mem_handle_t)] = {};
    char cmsgBuf[CMSG_SPACE(sizeof(ze_ipc_mem_handle_t))];

    struct iovec msgBuffer;
    msgBuffer.iov_base = sendBuf;
    msgBuffer.iov_len = sizeof(*sendBuf);

    struct msghdr msgHeader = {};
    msgHeader.msg_iov = &msgBuffer;
    msgHeader.msg_iovlen = 1;
    msgHeader.msg_control = cmsgBuf;
    msgHeader.msg_controllen = CMSG_LEN(sizeof(fd));

    struct cmsghdr *controlHeader = CMSG_FIRSTHDR(&msgHeader);
    controlHeader->cmsg_type = SCM_RIGHTS;
    controlHeader->cmsg_level = SOL_SOCKET;
    controlHeader->cmsg_len = CMSG_LEN(sizeof(fd));

    *(int *)CMSG_DATA(controlHeader) = fd;
    ssize_t bytesSent = sendmsg(socket, &msgHeader, 0);
    if (bytesSent < 0) {
        return -1;
    }

    return 0;
}

static int recvmsg_fd(int socket) {
    int fd = -1;
    char recvBuf[sizeof(ze_ipc_mem_handle_t)] = {};
    char cmsgBuf[CMSG_SPACE(sizeof(ze_ipc_mem_handle_t))];

    struct iovec msgBuffer;
    msgBuffer.iov_base = recvBuf;
    msgBuffer.iov_len = sizeof(recvBuf);

    struct msghdr msgHeader = {};
    msgHeader.msg_iov = &msgBuffer;
    msgHeader.msg_iovlen = 1;
    msgHeader.msg_control = cmsgBuf;
    msgHeader.msg_controllen = CMSG_LEN(sizeof(fd));

    ssize_t bytesSent = recvmsg(socket, &msgHeader, 0);
    if (bytesSent < 0) {
        return -1;
    }

    struct cmsghdr *controlHeader = CMSG_FIRSTHDR(&msgHeader);
    memmove(&fd, CMSG_DATA(controlHeader), sizeof(int));
    return fd;
}

void run_client(int commSocket, uint32_t clientId) {
    std::cout << "Client " << clientId << ", process ID: " << std::dec << getpid() << "\n";
    char *heapBuffer = new char[allocSize];
    for (size_t i = 0; i < allocSize; ++i) {
        heapBuffer[i] = static_cast<char>(i + 1);
    }
    // get the dma_buf from the other process
    int dma_buf_fd = recvmsg_fd(commSocket);
    if (dma_buf_fd < 0) {
        std::cerr << "Failing to get dma_buf fd from server\n";
        std::terminate();
    }

    ze_ipc_mem_handle_t pIpcHandle;
    memcpy(&pIpcHandle, static_cast<void *>(&dma_buf_fd), sizeof(dma_buf_fd));
   
    omp_interop_t o = 0;
    #pragma omp interop init(targetsync:  o) 

   void *zeIpcBuffer;
   int err = -1;
   ze_context_handle_t ze_context = static_cast<ze_context_handle_t>(omp_get_interop_ptr(o, omp_ipr_device_context, &err));
   assert (err >= 0 && "omp_get_interop_ptr(omp_ipr_device_context)");
   ze_device_handle_t ze_device =  static_cast<ze_device_handle_t>(omp_get_interop_ptr(o, omp_ipr_device, &err));
   assert (err >= 0 && "omp_get_interop_ptr(omp_ipr_device)");
   SUCCESS_OR_TERMINATE(zeMemOpenIpcHandle(ze_context, ze_device,
                                           pIpcHandle, 0u, &zeIpcBuffer));

#ifndef MANUAL_COPY
   err = omp_target_memcpy(zeIpcBuffer, heapBuffer, allocSize, 0,0, 0,omp_get_initial_device());
   assert (err >= 0 && "omp_target_memcpy");
#else
    char* zeIpcBuffer_char = static_cast<char *>(zeIpcBuffer);
    #pragma omp target is_device_ptr(zeIpcBuffer_char) map(to: heapBuffer[0:allocSize]) 
    for (size_t i = 0; i < allocSize; ++i) {
        zeIpcBuffer_char[i] = heapBuffer[i];
    }
#endif
   #pragma omp interop destroy(o) 

   SUCCESS_OR_TERMINATE(zeMemCloseIpcHandle(ze_context, zeIpcBuffer));
   std::cout << "end client" << std::endl;
   delete[] heapBuffer;
}

void run_server(bool &validRet) {
    std::cout << "Server process ID " << std::dec << getpid() << "\n";
    sycl::queue Q(sycl::gpu_selector{});
    sycl::context ctx = Q.get_context();

    void *zeBuffer = sycl::malloc_device(allocSize, Q);

    for (uint32_t i = 0; i < CHILDPROCESSES; i++) {
        // Initialize the IPC buffer
        int value = 3;
        Q.fill(zeBuffer, reinterpret_cast<void *>(&value), allocSize).wait();
        // Get a dma_buf for the previously allocated pointer
        ze_ipc_mem_handle_t pIpcHandle;
        SUCCESS_OR_TERMINATE(zeMemGetIpcHandle(ctx.get_native<sycl::backend::level_zero>(), 
                                               zeBuffer, &pIpcHandle));

        std::cout << "1" << std::endl;
        // Pass the dma_buf to the other process
        int dma_buf_fd;
        memcpy(static_cast<void *>(&dma_buf_fd), &pIpcHandle, sizeof(dma_buf_fd));
        int commSocket = sv[i][0];
        if (sendmsg_fd(commSocket, static_cast<int>(dma_buf_fd)) < 0) {
            std::cerr << "Failing to send dma_buf fd to client\n";
            std::terminate();
        }
        std::cout << "2" << std::endl;

        char *heapBuffer = new char[allocSize];
        for (size_t i = 0; i < allocSize; ++i) {
            heapBuffer[i] = static_cast<char>(i + 1);
        }
        std::cout << "3" << std::endl;
        // Wait for child to exit
        int child_status;
        pid_t clientPId = wait(&child_status);
        if (clientPId <= 0) {
            std::cerr << "Client terminated abruptly with error code " << strerror(errno) << "\n";
            std::terminate();
        }
        std::cout << "4" << std::endl;
        void *validateBuffer = sycl::malloc_shared(allocSize, Q);     
        
        value = 5;
        Q.fill(validateBuffer, reinterpret_cast<void *>(&value), allocSize).wait();

        // Copy from device-allocated memory
        Q.memcpy(validateBuffer, zeBuffer, allocSize).wait();
        // Validate stack and buffers have the original data from heapBuffer
        validRet = (0 == memcmp(heapBuffer, validateBuffer, allocSize));
        delete[] heapBuffer;
    }
}

int main(int argc, char *argv[]) {
    verbose = isVerbose(argc, argv);
    bool outputValidationSuccessful;

    for (uint32_t i = 0; i < CHILDPROCESSES; i++) {
        if (socketpair(PF_UNIX, SOCK_STREAM, 0, sv[i]) < 0) {
            perror("socketpair");
            exit(1);
        }
    }

    pid_t childPids[CHILDPROCESSES];
    for (uint32_t i = 0; i < CHILDPROCESSES; i++) {
        childPids[i] = fork();
        if (childPids[i] < 0) {
            perror("fork");
            exit(1);
        } else if (childPids[i] == 0) {
            close(sv[i][0]);
            run_client(sv[i][1], i);
            close(sv[i][1]);
            exit(0);
        }
    }

    run_server(outputValidationSuccessful);

    std::cout << "\nZello IPC Results validation "
              << (outputValidationSuccessful ? "PASSED" : "FAILED")
              << std::endl;

    return 0;
}
