#ifndef PUTER_DEVICE_CUDA_ATTENTION_ATTENTION_H
#define PUTER_DEVICE_CUDA_ATTENTION_ATTENTION_H

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

void cuda_attention_status_clear(CUDAStatus* status);

void cuda_attention_register_module_source(const char* source);

const char* cuda_attention_module_source(void);

int cuda_attention_kernel_name(
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
