#ifndef PUTER_DEVICE_CUDA_CONVOLUTION_CONV3D_H
#define PUTER_DEVICE_CUDA_CONVOLUTION_CONV3D_H

#include "convolution.h"

#ifdef __cplusplus
extern "C" {
#endif

int cuda_dispatch_conv3d(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef weightRef,
    CUDABufferRef biasRef,
    CUDABufferRef outRef,
    uint32_t batch,
    uint32_t inChannels,
    uint32_t inDepth,
    uint32_t inHeight,
    uint32_t inWidth,
    uint32_t outChannels,
    uint32_t kernelDepth,
    uint32_t kernelHeight,
    uint32_t kernelWidth,
    uint32_t outDepth,
    uint32_t outHeight,
    uint32_t outWidth,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
