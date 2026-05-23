#ifndef PUTER_DEVICE_METAL_CAUSAL_ADJUSTMENT_H
#define PUTER_DEVICE_METAL_CAUSAL_ADJUSTMENT_H

#include "causal.h"

#ifdef __cplusplus
extern "C" {
#endif

int metal_dispatch_backdoor_adjustment(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef conditionalRef,
    MetalBufferRef marginalRef,
    MetalBufferRef outRef,
    uint32_t xCount,
    uint32_t zCount,
    uint32_t yCount,
    uint64_t completionToken,
    MetalStatus* status
);

int metal_dispatch_frontdoor_adjustment(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef mediatorRef,
    MetalBufferRef outcomeRef,
    MetalBufferRef marginalRef,
    MetalBufferRef outRef,
    uint32_t xCount,
    uint32_t mCount,
    uint32_t yCount,
    uint64_t completionToken,
    MetalStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
