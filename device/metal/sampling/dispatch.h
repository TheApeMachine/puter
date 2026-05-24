#ifndef PUTER_DEVICE_METAL_SAMPLING_DISPATCH_H
#define PUTER_DEVICE_METAL_SAMPLING_DISPATCH_H

#include "sampling.h"

#ifdef __cplusplus
extern "C" {
#endif

int metal_dispatch_sampling(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef logitsRef,
    MetalBufferRef scoresRef,
    MetalBufferRef indicesRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint32_t paddedCount,
    float target,
    uint64_t completionToken,
    MetalStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
