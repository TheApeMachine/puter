#ifndef PUTER_DEVICE_CUDA_ROPE_ROTATE_H
#define PUTER_DEVICE_CUDA_ROPE_ROTATE_H

#include "rope.h"

#ifdef __cplusplus
extern "C" {
#endif

int cuda_dispatch_rope(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef outRef,
    uint32_t seqLen,
    uint32_t numHeads,
    uint32_t headDim,
    uint32_t pairCount,
    float theta,
    float ropeFactor,
    float lowFreqFactor,
    float highFreqFactor,
    uint32_t originalContext,
    uint32_t halfMode,
    uint32_t positionOffset,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_rope_pairs(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef outRef,
    CUDABufferRef cosRef,
    CUDABufferRef sinRef,
    uint32_t halfDim,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
