#ifndef PUTER_DEVICE_CUDA_ROPE_H
#define PUTER_DEVICE_CUDA_ROPE_H

#include "../internal/bridge/core.h"

#ifdef __cplusplus
extern "C" {
#endif

void cuda_rope_register_module_source(const char* source);

const char* cuda_rope_module_source(void);

void cuda_rope_status_clear(CUDAStatus* status);

void cuda_rope_status_set(CUDAStatus* status, int code, const char* message);

int cuda_rope_kernel_name(
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
