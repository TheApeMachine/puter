#ifndef PUTER_DEVICE_METAL_REDUCTION_AGGREGATE_H
#define PUTER_DEVICE_METAL_REDUCTION_AGGREGATE_H

#include "reduction.h"

#ifdef __cplusplus
extern "C" {
#endif

int metal_dispatch_reduction(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef scratchARef,
    MetalBufferRef scratchBRef,
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
