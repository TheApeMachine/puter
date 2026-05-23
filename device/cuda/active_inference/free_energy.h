#ifndef PUTER_DEVICE_CUDA_ACTIVE_INFERENCE_FREE_ENERGY_H
#define PUTER_DEVICE_CUDA_ACTIVE_INFERENCE_FREE_ENERGY_H

#include "active_inference.h"

#ifdef __cplusplus
extern "C" {
#endif

int cuda_dispatch_free_energy(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef likelihoodRef,
    CUDABufferRef posteriorRef,
    CUDABufferRef priorRef,
    CUDABufferRef scratchRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint32_t partialCount,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_expected_free_energy(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef predictedObsRef,
    CUDABufferRef preferredObsRef,
    CUDABufferRef predictedStateRef,
    CUDABufferRef scratchRef,
    CUDABufferRef outRef,
    uint32_t obsCount,
    uint32_t stateCount,
    uint32_t obsPartialCount,
    uint32_t statePartialCount,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
