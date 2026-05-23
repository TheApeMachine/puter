#ifndef PUTER_DEVICE_CUDA_MATMUL_PRODUCT_H
#define PUTER_DEVICE_CUDA_MATMUL_PRODUCT_H

#include "matmul.h"

#ifdef __cplusplus
extern "C" {
#endif

int cuda_dispatch_matmul(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef leftRef,
    CUDABufferRef rightRef,
    CUDABufferRef outRef,
    uint32_t rows,
    uint32_t inner,
    uint32_t cols,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_matmul_add(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef leftRef,
    CUDABufferRef rightRef,
    CUDABufferRef biasRef,
    CUDABufferRef outRef,
    uint32_t rows,
    uint32_t inner,
    uint32_t cols,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
