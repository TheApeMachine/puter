#ifndef PUTER_DEVICE_METAL_ACTIVATION_LUT_H
#define PUTER_DEVICE_METAL_ACTIVATION_LUT_H

#include "activation.h"

#ifdef __cplusplus
extern "C" {
#endif

int metal_dispatch_lut_gather(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    MetalBufferRef lutRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
