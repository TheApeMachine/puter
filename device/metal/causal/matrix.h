#ifndef PUTER_DEVICE_METAL_CAUSAL_MATRIX_H
#define PUTER_DEVICE_METAL_CAUSAL_MATRIX_H

#include "causal.h"

#ifdef __cplusplus
extern "C" {
#endif

int metal_dispatch_inv_sqrt_dim_scale(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef dimRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);

int metal_dispatch_logsumexp(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint64_t completionToken,
    MetalStatus* status
);

int metal_dispatch_outer(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef leftRef,
    MetalBufferRef rightRef,
    MetalBufferRef outRef,
    uint32_t rows,
    uint32_t cols,
    uint64_t completionToken,
    MetalStatus* status
);

int metal_dispatch_fma_float32(
    MetalDeviceRef contextRef,
    MetalBufferRef aRef,
    MetalBufferRef bRef,
    MetalBufferRef cRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);

int metal_dispatch_inv_std_dev_float32(
    MetalDeviceRef contextRef,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);

int metal_dispatch_unary_named_float32(
    MetalDeviceRef contextRef,
    const char* kernelName,
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
