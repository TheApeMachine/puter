#ifndef PUTER_DEVICE_METAL_CAUSAL_INTERVENTION_H
#define PUTER_DEVICE_METAL_CAUSAL_INTERVENTION_H

#include "causal.h"

#ifdef __cplusplus
extern "C" {
#endif

int metal_dispatch_do_intervene(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef adjacencyRef,
    MetalBufferRef intervenedRef,
    MetalBufferRef outRef,
    uint32_t nodeCount,
    uint32_t intervenedCount,
    uint64_t completionToken,
    MetalStatus* status
);

int metal_dispatch_cate(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef treatedRef,
    MetalBufferRef controlRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);

int metal_dispatch_counterfactual(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef observedYRef,
    MetalBufferRef observedXRef,
    MetalBufferRef counterfactualXRef,
    MetalBufferRef slopeRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);

int metal_dispatch_iv_estimate(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef instrumentRef,
    MetalBufferRef treatmentRef,
    MetalBufferRef outcomeRef,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint32_t partialCount,
    uint64_t completionToken,
    MetalStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
