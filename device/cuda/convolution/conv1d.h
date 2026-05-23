#ifndef PUTER_DEVICE_CUDA_CONVOLUTION_CONV1D_H
#define PUTER_DEVICE_CUDA_CONVOLUTION_CONV1D_H

#include "convolution.h"

#ifdef __cplusplus
extern "C" {
#endif

int cuda_dispatch_conv1d(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef weightRef,
    CUDABufferRef biasRef,
    CUDABufferRef outRef,
    uint32_t batch,
    uint32_t inChannels,
    uint32_t inLength,
    uint32_t outChannels,
    uint32_t kernelLength,
    uint32_t outLength,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
