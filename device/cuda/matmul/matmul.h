#ifndef PUTER_DEVICE_CUDA_MATMUL_H
#define PUTER_DEVICE_CUDA_MATMUL_H

#include "../internal/bridge/core.h"

#ifdef __cplusplus
extern "C" {
#endif

void cuda_matmul_status_clear(CUDAStatus* status);

void cuda_matmul_status_set(CUDAStatus* status, int code, const char* message);

void cuda_matmul_register_module_source(const char* source);

const char* cuda_matmul_module_source(void);

int cuda_matmul_kernel_name(
    char* out,
    size_t outBytes,
    const char* operationName,
    int elementDType,
    CUDAStatus* status
);

int cuda_matmul_dispatch_tiled(
    CUDADeviceRef contextRef,
    const char* operationName,
    int elementDType,
    CUDABufferRef leftRef,
    CUDABufferRef rightRef,
    CUDABufferRef biasRef,
    CUDABufferRef outRef,
    uint32_t rows,
    uint32_t inner,
    uint32_t cols,
    int hasBias,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
