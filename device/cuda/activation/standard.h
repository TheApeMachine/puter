#ifndef PUTER_DEVICE_CUDA_ACTIVATION_STANDARD_H
#define PUTER_DEVICE_CUDA_ACTIVATION_STANDARD_H

#include "activation.h"

#ifdef __cplusplus
extern "C" {
#endif

int cuda_dispatch_unary_elementwise(
    CUDADeviceRef contextRef,
    int operation,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
