#ifndef PUTER_DEVICE_CUDA_REDUCTION_AGGREGATE_H
#define PUTER_DEVICE_CUDA_REDUCTION_AGGREGATE_H

#include "reduction.h"

#ifdef __cplusplus
extern "C" {
#endif

int cuda_dispatch_reduction(
    CUDADeviceRef contextRef,
    int operation,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef scratchARef,
    CUDABufferRef scratchBRef,
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
