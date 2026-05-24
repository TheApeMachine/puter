#ifndef PUTER_DEVICE_METAL_LOSSES_DISPATCH_H
#define PUTER_DEVICE_METAL_LOSSES_DISPATCH_H

#include "losses.h"

#ifdef __cplusplus
extern "C" {
#endif

int metal_dispatch_pair_loss(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef predictionsRef,
    MetalBufferRef targetsRef,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint32_t partialCount,
    uint64_t completionToken,
    MetalStatus* status
);

int metal_dispatch_cross_entropy_loss(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef logitsRef,
    MetalBufferRef targetsRef,
    MetalBufferRef scratchRef,
    MetalBufferRef outRef,
    uint32_t batch,
    uint32_t classes,
    uint64_t completionToken,
    MetalStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
