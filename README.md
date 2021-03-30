# Using IPC handle for different programing model

# Sycl + level 0

```
dpcpp -lze_loader syclo_ipc_copy_dma_buf.cpp -o syclo_ipc_copy_dma_buf
SYCL_DEVICE_FILTER=PI_LEVEL_ZERO ./syclo_ipc_copy_dma_buf
```

# Level 0
From: `https://github.com/intel/compute-runtime/blob/master/level_zero/core/test/black_box_tests/zello_ipc_copy_dma_buf.cpp`
```
g++ -lze_loader zello_ipc_copy_dma_buf.cpp -o zello_ipc_copy_dma_buf
./zello_ipc_copy_dma_buf
```


# NVIDIA
