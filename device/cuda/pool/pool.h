#ifndef PUTER_DEVICE_CUDA_POOL_H
#define PUTER_DEVICE_CUDA_POOL_H

#include "../internal/bridge/core.h"

#ifdef __cplusplus
extern "C" {
#endif

void cuda_vision_status_set(CUDAStatus* status, int code, const char* message);

int cuda_vision_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
