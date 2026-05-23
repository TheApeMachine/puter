#ifndef PUTER_DEVICE_METAL_ATTENTION_MASKING_H
#define PUTER_DEVICE_METAL_ATTENTION_MASKING_H

#include "attention.h"

#ifdef __cplusplus
extern "C" {
#endif

int metal_dispatch_apply_mask(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef maskRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);

int metal_dispatch_causal_mask(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint64_t completionToken,
    MetalStatus* status
);

int metal_dispatch_alibi_bias(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef scoresRef,
    MetalBufferRef slopeRef,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint64_t completionToken,
    MetalStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
