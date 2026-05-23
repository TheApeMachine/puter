#ifndef PUTER_DEVICE_CUDA_CAUSAL_MATRIX_H
#define PUTER_DEVICE_CUDA_CAUSAL_MATRIX_H

#include "causal.h"

#ifdef __cplusplus
extern "C" {
#endif

int cuda_dispatch_inv_sqrt_dim_scale(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef dimRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_logsumexp(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_outer(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef leftRef,
    CUDABufferRef rightRef,
    CUDABufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_fma_float32(
    CUDADeviceRef contextRef,
    CUDABufferRef aRef,
    CUDABufferRef bRef,
    CUDABufferRef cRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_inv_std_dev_float32(
    CUDADeviceRef contextRef,
    CUDABufferRef inputRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_unary_named_float32(
    CUDADeviceRef contextRef,
    const char* kernelName,
    CUDABufferRef inputRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
