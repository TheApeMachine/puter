#ifndef PUTER_DEVICE_CUDA_NORMALIZATION_H
#define PUTER_DEVICE_CUDA_NORMALIZATION_H

#include "../internal/bridge/core.h"

#ifdef __cplusplus
extern "C" {
#endif

void cuda_normalization_status_clear(CUDAStatus* status);

void cuda_normalization_status_set(CUDAStatus* status, int code, const char* message);

void cuda_normalization_register_module_source(const char* source);

const char* cuda_normalization_module_source(void);

int cuda_normalization_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    CUDAStatus* status
);

int cuda_normalization_dispatch_rows(
    CUDADeviceRef contextRef,
    const char* operationName,
    int elementDType,
    void** bufferRefs,
    size_t bufferCount,
    void** uintArgs,
    size_t uintArgCount,
    uint32_t rows,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
