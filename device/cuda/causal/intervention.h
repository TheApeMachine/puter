#ifndef PUTER_DEVICE_CUDA_CAUSAL_INTERVENTION_H
#define PUTER_DEVICE_CUDA_CAUSAL_INTERVENTION_H

#include "causal.h"

#ifdef __cplusplus
extern "C" {
#endif

int cuda_dispatch_do_intervene(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef adjacencyRef,
    CUDABufferRef intervenedRef,
    CUDABufferRef outRef,
    uint32_t nodeCount,
    uint32_t intervenedCount,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_cate(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef treatedRef,
    CUDABufferRef controlRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_counterfactual(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef observedYRef,
    CUDABufferRef observedXRef,
    CUDABufferRef counterfactualXRef,
    CUDABufferRef slopeRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_iv_estimate(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef instrumentRef,
    CUDABufferRef treatmentRef,
    CUDABufferRef outcomeRef,
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
