#ifndef PUTER_DEVICE_CUDA_POOL_ADAPTIVE_H
#define PUTER_DEVICE_CUDA_POOL_ADAPTIVE_H

#include "pool.h"

#ifdef __cplusplus
extern "C" {
#endif

int cuda_dispatch_adaptive_max_pool2d(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef outRef,
    uint32_t batch,
    uint32_t channels,
    uint32_t inHeight,
    uint32_t inWidth,
    uint32_t outHeight,
    uint32_t outWidth,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_adaptive_avg_pool2d(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef outRef,
    uint32_t batch,
    uint32_t channels,
    uint32_t inHeight,
    uint32_t inWidth,
    uint32_t outHeight,
    uint32_t outWidth,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
