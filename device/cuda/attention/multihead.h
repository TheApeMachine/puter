#ifndef PUTER_DEVICE_CUDA_ATTENTION_MULTIHEAD_H
#define PUTER_DEVICE_CUDA_ATTENTION_MULTIHEAD_H

#include "attention.h"

#ifdef __cplusplus
extern "C" {
#endif

int cuda_dispatch_multi_head_attention(
    CUDADeviceRef contextRef,
    int elementDType,
    int variant,
    CUDABufferRef queryRef,
    CUDABufferRef keyRef,
    CUDABufferRef valueRef,
    CUDABufferRef outRef,
    uint32_t seqQ,
    uint32_t seqK,
    uint32_t numHeads,
    uint32_t kvHeads,
    uint32_t headDim,
    uint32_t windowSize,
    uint32_t causal,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
