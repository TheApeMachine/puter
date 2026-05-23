#ifndef PUTER_DEVICE_CUDA_ELEMENTWISE_H
#define PUTER_DEVICE_CUDA_ELEMENTWISE_H

#include "../internal/bridge/core.h"

#ifdef __cplusplus
extern "C" {
#endif

void cuda_elementwise_status_clear(CUDAStatus* status);

void cuda_elementwise_status_set(CUDAStatus* status, int code, const char* message);

const char* cuda_elementwise_element_dtype_suffix(int elementDType);

int cuda_elementwise_compose_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    const char* dtypeSuffix,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
