#ifndef CARAMBA_BACKEND_DEVICE_METAL_BRIDGE_VISION_PRIVATE_H
#define CARAMBA_BACKEND_DEVICE_METAL_BRIDGE_VISION_PRIVATE_H

#include "../internal/bridge/core_private.h"

#include <Foundation/Foundation.h>
#include <CUDA/CUDA.h>

typedef void (^CUDAVisionEncodeBlock)(id<MTLComputeCommandEncoder> encoder);

void cuda_vision_status_set(CUDAStatus* status, int code, const char* message);

int cuda_vision_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    CUDAStatus* status
);

int cuda_vision_dispatch(
    CUDADeviceRef contextRef,
    const char* kernelName,
    NSUInteger threadCount,
    uint64_t completionToken,
    CUDAStatus* status,
    CUDAVisionEncodeBlock encode
);

#endif
