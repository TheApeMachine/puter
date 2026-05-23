#ifndef PUTER_DEVICE_CUDA_CAUSAL_ADJUSTMENT_H
#define PUTER_DEVICE_CUDA_CAUSAL_ADJUSTMENT_H

#include "causal.h"

#ifdef __cplusplus
extern "C" {
#endif

int cuda_dispatch_backdoor_adjustment(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef conditionalRef,
    CUDABufferRef marginalRef,
    CUDABufferRef outRef,
    uint32_t xCount,
    uint32_t zCount,
    uint32_t yCount,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_frontdoor_adjustment(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef mediatorRef,
    CUDABufferRef outcomeRef,
    CUDABufferRef marginalRef,
    CUDABufferRef outRef,
    uint32_t xCount,
    uint32_t mCount,
    uint32_t yCount,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
