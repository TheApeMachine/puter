#ifndef PUTER_DEVICE_METAL_ROPE_ROTATE_H
#define PUTER_DEVICE_METAL_ROPE_ROTATE_H

#include "rope.h"

#ifdef __cplusplus
extern "C" {
#endif

int metal_dispatch_rope(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t seqLen,
    uint32_t numHeads,
    uint32_t headDim,
    uint32_t pairCount,
    float theta,
    float ropeFactor,
    float lowFreqFactor,
    float highFreqFactor,
    uint32_t originalContext,
    uint32_t halfMode,
    uint32_t positionOffset,
    uint64_t completionToken,
    MetalStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
