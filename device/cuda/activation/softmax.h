#ifndef PUTER_DEVICE_CUDA_ACTIVATION_SOFTMAX_H
#define PUTER_DEVICE_CUDA_ACTIVATION_SOFTMAX_H

#include "activation.h"

#ifdef __cplusplus
extern "C" {
#endif

int cuda_dispatch_softmax(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef inputRef,
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
