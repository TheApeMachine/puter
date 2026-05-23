#ifndef PUTER_DEVICE_METAL_MATMUL_PRODUCT_H
#define PUTER_DEVICE_METAL_MATMUL_PRODUCT_H

#include "matmul.h"

#ifdef __cplusplus
extern "C" {
#endif

int metal_dispatch_matmul(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef leftRef,
    MetalBufferRef rightRef,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t inner,
    uint32_t cols,
    uint64_t completionToken,
    MetalStatus* status
);

int metal_dispatch_matmul_add(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef leftRef,
    MetalBufferRef rightRef,
    MetalBufferRef biasRef,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t inner,
    uint32_t cols,
    uint64_t completionToken,
    MetalStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
