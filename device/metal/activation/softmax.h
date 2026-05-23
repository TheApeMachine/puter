#ifndef PUTER_DEVICE_METAL_ACTIVATION_SOFTMAX_H
#define PUTER_DEVICE_METAL_ACTIVATION_SOFTMAX_H

#include "activation.h"

#ifdef __cplusplus
extern "C" {
#endif

int metal_dispatch_softmax(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint64_t completionToken,
    MetalStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
