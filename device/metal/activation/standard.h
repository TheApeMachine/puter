#ifndef PUTER_DEVICE_METAL_ACTIVATION_STANDARD_H
#define PUTER_DEVICE_METAL_ACTIVATION_STANDARD_H

#include "activation.h"

#ifdef __cplusplus
extern "C" {
#endif

int metal_dispatch_unary_elementwise(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
