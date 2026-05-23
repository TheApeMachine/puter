#ifndef PUTER_DEVICE_CUDA_DOT_H
#define PUTER_DEVICE_CUDA_DOT_H

#include "../internal/bridge/core.h"

#ifdef __cplusplus
extern "C" {
#endif

void cuda_dot_status_clear(CUDAStatus* status);

void cuda_dot_status_set(CUDAStatus* status, int code, const char* message);

void cuda_dot_register_module_source(const char* source);

const char* cuda_dot_module_source(void);

int cuda_dot_kernel_name(
    char* out,
    size_t outBytes,
    const char* phase,
    int elementDType,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
