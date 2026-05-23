#ifndef PUTER_DEVICE_METAL_ATTENTION_FLASH_H
#define PUTER_DEVICE_METAL_ATTENTION_FLASH_H

#include "attention.h"

#ifdef __cplusplus
extern "C" {
#endif

int metal_dispatch_flash_attention(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef queryRef,
    MetalBufferRef keyRef,
    MetalBufferRef valueRef,
    MetalBufferRef outRef,
    uint32_t seqQ,
    uint32_t seqK,
    uint32_t depth,
    uint32_t valueDim,
    uint64_t completionToken,
    MetalStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
