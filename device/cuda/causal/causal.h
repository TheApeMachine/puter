#ifndef PUTER_DEVICE_CUDA_CAUSAL_CAUSAL_H
#define PUTER_DEVICE_CUDA_CAUSAL_CAUSAL_H

#include "../internal/bridge/core.h"

#ifdef __cplusplus
extern "C" {
#endif

void cuda_causal_status_set(CUDAStatus* status, int code, const char* message);

int cuda_causal_kernel_name(
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
