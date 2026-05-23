#ifndef PUTER_DEVICE_METAL_ELEMENTWISE_AXPY_H
#define PUTER_DEVICE_METAL_ELEMENTWISE_AXPY_H

#include "elementwise.h"

#ifdef __cplusplus
extern "C" {
#endif

int metal_dispatch_axpy(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef yRef,
    MetalBufferRef xRef,
    float alpha,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
