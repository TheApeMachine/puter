#ifndef PUTER_DEVICE_CUDA_ACTIVE_INFERENCE_H
#define PUTER_DEVICE_CUDA_ACTIVE_INFERENCE_H

#include "../internal/bridge/core.h"

#ifdef __cplusplus
extern "C" {
#endif

void cuda_active_register_module_source(const char* source);

const char* cuda_active_module_source(void);

void cuda_active_status_clear(CUDAStatus* status);

void cuda_active_status_set(CUDAStatus* status, int code, const char* message);

int cuda_active_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    const char* phase,
    int elementDType,
    CUDAStatus* status
);

int cuda_active_single_kernel_name(
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
