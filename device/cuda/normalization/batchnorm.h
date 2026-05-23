#ifndef PUTER_DEVICE_CUDA_NORMALIZATION_BATCHNORM_H
#define PUTER_DEVICE_CUDA_NORMALIZATION_BATCHNORM_H

#include "normalization.h"

#ifdef __cplusplus
extern "C" {
#endif

int cuda_dispatch_batchnorm_eval(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef scaleRef,
    CUDABufferRef biasRef,
    CUDABufferRef meanRef,
    CUDABufferRef varianceRef,
    CUDABufferRef outputRef,
    uint32_t batch,
    uint32_t channels,
    uint32_t spatial,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
