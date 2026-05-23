#ifndef PUTER_DEVICE_CUDA_REDUCTION_H
#define PUTER_DEVICE_CUDA_REDUCTION_H

#include "../internal/bridge/core.h"

#ifdef __cplusplus
extern "C" {
#endif

void cuda_reduction_status_clear(CUDAStatus* status);

void cuda_reduction_status_set(CUDAStatus* status, int code, const char* message);

void cuda_reduction_register_module_source(const char* source);

const char* cuda_reduction_module_source(void);

int cuda_reduction_kernel_name(
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
