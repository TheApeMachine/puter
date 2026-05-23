#ifndef PUTER_DEVICE_CUDA_EMBEDDING_H
#define PUTER_DEVICE_CUDA_EMBEDDING_H

#include "../internal/bridge/core.h"

#ifdef __cplusplus
extern "C" {
#endif

void cuda_transformer_status_set(CUDAStatus* status, int code, const char* message);

void cuda_embedding_register_module_source(const char* source);

const char* cuda_embedding_module_source(void);

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
