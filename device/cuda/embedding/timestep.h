#ifndef PUTER_DEVICE_CUDA_EMBEDDING_TIMESTEP_H
#define PUTER_DEVICE_CUDA_EMBEDDING_TIMESTEP_H

#include "embedding.h"

#ifdef __cplusplus
extern "C" {
#endif

int cuda_dispatch_timestep_embedding(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef timestepsRef,
    CUDABufferRef maxPeriodRef,
    CUDABufferRef downscaleRef,
    CUDABufferRef flipRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint32_t dim,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
