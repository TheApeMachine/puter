#include "attention.h"
#include "../internal/bridge/core_private.h"

#include <stdio.h>

static const char* g_cuda_attention_module_source = NULL;

void cuda_attention_register_module_source(const char* source) {
    g_cuda_attention_module_source = source;
}

const char* cuda_attention_module_source(void) {
    return g_cuda_attention_module_source;
}

void cuda_attention_status_clear(CUDAStatus* status) {
    cuda_status_clear(status);
}

void cuda_transformer_status_set(CUDAStatus* status, int code, const char* message) {
    cuda_status_set(status, code, message);
}

int cuda_transformer_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    CUDAStatus* status
) {
    const char* suffix = cuda_element_dtype_suffix(elementDType);

    if (suffix == NULL) {
        cuda_transformer_status_set(status, -6, "unknown CUDA attention dtype");
        return -6;
    }

    return cuda_compose_kernel_name(out, outBytes, operationName, suffix, status);
}

int cuda_attention_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    CUDAStatus* status
) {
    return cuda_transformer_kernel_name(out, outBytes, operationName, elementDType, status);
}
