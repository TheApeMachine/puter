#ifndef PUTER_DEVICE_CUDA_ACTIVATION_ACTIVATION_H
#define PUTER_DEVICE_CUDA_ACTIVATION_ACTIVATION_H

#include "../internal/bridge/core.h"

#ifdef __cplusplus
extern "C" {
#endif

void cuda_activation_register_module_source(const char* source);

const char* cuda_activation_module_source(void);

void cuda_activation_status_clear(CUDAStatus* status);

void cuda_activation_status_set(CUDAStatus* status, int code, const char* message);

const char* cuda_activation_element_dtype_suffix(int elementDType);

int cuda_activation_compose_kernel_name(
    char* out,
    size_t outBytes,
    const char* prefix,
    const char* suffix,
    CUDAStatus* status
);

uint32_t cuda_activation_vector_launch_count(uint32_t count, int elementDType);

#ifdef __cplusplus
}
#endif

#endif
