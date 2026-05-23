#ifndef PUTER_DEVICE_CUDA_ATTENTION_SCALED_DOT_PRODUCT_H
#define PUTER_DEVICE_CUDA_ATTENTION_SCALED_DOT_PRODUCT_H

#include "attention.h"

#ifdef __cplusplus
extern "C" {
#endif

int cuda_dispatch_attention(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef queryRef,
    CUDABufferRef keyRef,
    CUDABufferRef valueRef,
    CUDABufferRef scoresRef,
    CUDABufferRef outRef,
    uint32_t seqQ,
    uint32_t seqK,
    uint32_t depth,
    uint32_t valueDim,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
