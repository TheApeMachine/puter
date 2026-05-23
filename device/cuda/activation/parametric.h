#ifndef PUTER_DEVICE_CUDA_ACTIVATION_PARAMETRIC_H
#define PUTER_DEVICE_CUDA_ACTIVATION_PARAMETRIC_H

#include "activation.h"

#ifdef __cplusplus
extern "C" {
#endif

int cuda_dispatch_unary_param(
    CUDADeviceRef contextRef,
    const char* operationPrefix,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef outRef,
    uint32_t count,
    float param,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_indexed_param(
    CUDADeviceRef contextRef,
    const char* operationPrefix,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef slopesRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
