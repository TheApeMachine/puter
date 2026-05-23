#ifndef PUTER_DEVICE_CUDA_POOL_H
#define PUTER_DEVICE_CUDA_POOL_H

#include "../internal/bridge/core.h"

#ifdef __cplusplus
extern "C" {
#endif

void cuda_vision_register_module_source(const char* source);

const char* cuda_vision_module_source(void);

void cuda_vision_status_set(CUDAStatus* status, int code, const char* message);

int cuda_vision_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    CUDAStatus* status
);

int cuda_vision_dispatch_pool2d(
    CUDADeviceRef contextRef,
    const char* operationName,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef outRef,
    uint32_t batch,
    uint32_t channels,
    uint32_t inHeight,
    uint32_t inWidth,
    uint32_t outHeight,
    uint32_t outWidth,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
