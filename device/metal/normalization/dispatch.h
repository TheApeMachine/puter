#ifndef PUTER_DEVICE_METAL_NORMALIZATION_DISPATCH_H
#define PUTER_DEVICE_METAL_NORMALIZATION_DISPATCH_H

#include "../internal/bridge/core.h"

#ifdef __cplusplus
extern "C" {
#endif

int metal_dispatch_groupnorm(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef scaleRef,
    MetalBufferRef biasRef,
    MetalBufferRef outRef,
    uint32_t batch,
    uint32_t channels,
    uint32_t spatial,
    uint32_t groups,
    uint64_t completionToken,
    MetalStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
