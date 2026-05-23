#ifndef PUTER_DEVICE_METAL_ACTIVATION_PARAMETRIC_H
#define PUTER_DEVICE_METAL_ACTIVATION_PARAMETRIC_H

#include "activation.h"

#ifdef __cplusplus
extern "C" {
#endif

int metal_dispatch_unary_param(
    MetalDeviceRef contextRef,
    const char* kernelName,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t count,
    float param,
    uint64_t completionToken,
    MetalStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
