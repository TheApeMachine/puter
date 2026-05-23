#ifndef PUTER_DEVICE_CUDA_DROPOUT_MASK_H
#define PUTER_DEVICE_CUDA_DROPOUT_MASK_H

#include "dropout.h"

#ifdef __cplusplus
extern "C" {
#endif

int cuda_dispatch_dropout(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef outRef,
    uint32_t count,
    float scale,
    uint32_t threshold,
    uint32_t seedX,
    uint32_t seedY,
    uint32_t seedZ,
    uint32_t seedW,
    int identity,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
