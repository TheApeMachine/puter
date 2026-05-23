#ifndef PUTER_DEVICE_CUDA_ROPE_H
#define PUTER_DEVICE_CUDA_ROPE_H

#include "../internal/bridge/core.h"

#ifdef __cplusplus
extern "C" {
#endif

void cuda_transformer_status_set(CUDAStatus* status, int code, const char* message);

int cuda_transformer_kernel_name(
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
