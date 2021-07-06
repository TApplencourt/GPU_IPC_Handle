#include "zello_common.h"

#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <CL/sycl.hpp>
// One should read the manual
// https://github.com/smaslov-intel/llvm/blob/master/sycl/doc/extensions/LevelZeroBackend/LevelZeroBackend.md
#include "level_zero/ze_api.h"
#include <CL/sycl/backend/level_zero.hpp>

#define CHILDPROCESSES 4

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
    sycl::queue Q(sycl::gpu_selector{});
    
    sycl::device dev = Q.get_device();
    sycl::context ctx = Q.get_context();
    void *zeIpcBuffer;
    SUCCESS_OR_TERMINATE(zeMemOpenIpcHandle(ctx.get_native<sycl::backend::level_zero>(),
                                            dev.get_native<sycl::backend::level_zero>(), 
                                            pIpcHandle, 0u, &zeIpcBuffer));
    Q.memcpy(zeIpcBuffer, heapBuffer, allocSize).wait();
    SUCCESS_OR_TERMINATE(zeMemCloseIpcHandle(ctx.get_native<sycl::backend::level_zero>(), zeIpcBuffer));
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
        // Pass the dma_buf to the other process
        int dma_buf_fd;
        memcpy(static_cast<void *>(&dma_buf_fd), &pIpcHandle, sizeof(dma_buf_fd));
        int commSocket = sv[i][0];
        if (sendmsg_fd(commSocket, static_cast<int>(dma_buf_fd)) < 0) {
            std::cerr << "Failing to send dma_buf fd to client\n";
            std::terminate();
        }

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
