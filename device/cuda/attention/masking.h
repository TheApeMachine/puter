#ifndef PUTER_DEVICE_CUDA_ATTENTION_MASKING_H
#define PUTER_DEVICE_CUDA_ATTENTION_MASKING_H

#include "attention.h"

#ifdef __cplusplus
extern "C" {
#endif

int cuda_dispatch_apply_mask(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef maskRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_causal_mask(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_alibi_bias(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef scoresRef,
    CUDABufferRef slopeRef,
    CUDABufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
