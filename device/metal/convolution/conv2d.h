#ifndef PUTER_DEVICE_METAL_CONVOLUTION_CONV2D_H
#define PUTER_DEVICE_METAL_CONVOLUTION_CONV2D_H

#include "convolution.h"

#ifdef __cplusplus
extern "C" {
#endif

int metal_dispatch_conv2d(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef weightRef,
    MetalBufferRef biasRef,
    MetalBufferRef outRef,
    uint32_t batch,
    uint32_t inChannels,
    uint32_t inHeight,
    uint32_t inWidth,
    uint32_t outChannels,
    uint32_t kernelHeight,
    uint32_t kernelWidth,
    uint32_t outHeight,
    uint32_t outWidth,
    uint32_t strideHeight,
    uint32_t strideWidth,
    uint32_t paddingHeight,
    uint32_t paddingWidth,
    uint32_t dilationHeight,
    uint32_t dilationWidth,
    uint64_t completionToken,
    MetalStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
