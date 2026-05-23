#ifndef PUTER_DEVICE_CUDA_DOT_INNER_PRODUCT_H
#define PUTER_DEVICE_CUDA_DOT_INNER_PRODUCT_H

#include "dot.h"

#ifdef __cplusplus
extern "C" {
#endif

int cuda_dispatch_dot(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef leftRef,
    CUDABufferRef rightRef,
    CUDABufferRef scratchRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint32_t partialCount,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
