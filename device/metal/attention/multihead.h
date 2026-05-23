#ifndef PUTER_DEVICE_METAL_ATTENTION_MULTIHEAD_H
#define PUTER_DEVICE_METAL_ATTENTION_MULTIHEAD_H

#include "attention.h"

#ifdef __cplusplus
extern "C" {
#endif

int metal_dispatch_multi_head_attention(
    MetalDeviceRef contextRef,
    int elementDType,
    int variant,
    MetalBufferRef queryRef,
    MetalBufferRef keyRef,
    MetalBufferRef valueRef,
    MetalBufferRef outRef,
    uint32_t seqQ,
    uint32_t seqK,
    uint32_t numHeads,
    uint32_t kvHeads,
    uint32_t headDim,
    uint32_t windowSize,
    uint32_t causal,
    uint64_t completionToken,
    MetalStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
