#ifndef PUTER_DEVICE_CUDA_ACTIVE_INFERENCE_BELIEF_H
#define PUTER_DEVICE_CUDA_ACTIVE_INFERENCE_BELIEF_H

#include "active_inference.h"

#ifdef __cplusplus
extern "C" {
#endif

int cuda_dispatch_belief_update(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef likelihoodRef,
    CUDABufferRef priorRef,
    CUDABufferRef scratchRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint32_t partialCount,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_precision_weight(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef errorsRef,
    CUDABufferRef precisionRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
