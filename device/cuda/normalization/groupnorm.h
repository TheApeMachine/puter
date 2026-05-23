#ifndef PUTER_DEVICE_CUDA_NORMALIZATION_GROUPNORM_H
#define PUTER_DEVICE_CUDA_NORMALIZATION_GROUPNORM_H

#include "normalization.h"

#ifdef __cplusplus
extern "C" {
#endif

int cuda_dispatch_groupnorm(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef scaleRef,
    CUDABufferRef biasRef,
    CUDABufferRef outputRef,
    uint32_t batch,
    uint32_t channels,
    uint32_t spatial,
    uint32_t groups,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
