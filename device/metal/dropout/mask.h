#ifndef PUTER_DEVICE_METAL_DROPOUT_MASK_H
#define PUTER_DEVICE_METAL_DROPOUT_MASK_H

#include "dropout.h"

#ifdef __cplusplus
extern "C" {
#endif

int metal_dispatch_dropout(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t count,
    float scale,
    uint32_t threshold,
    uint32_t seed0,
    uint32_t seed1,
    uint32_t seed2,
    uint32_t seed3,
    uint64_t completionToken,
    MetalStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
