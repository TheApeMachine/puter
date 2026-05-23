#ifndef PUTER_DEVICE_METAL_ACTIVATION_GATED_H
#define PUTER_DEVICE_METAL_ACTIVATION_GATED_H

#include "activation.h"

#ifdef __cplusplus
extern "C" {
#endif

int metal_dispatch_swiglu(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef destinationRef,
    MetalBufferRef gateRef,
    MetalBufferRef upRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);

int metal_dispatch_swiglu_packed(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef destinationRef,
    MetalBufferRef packedRef,
    uint32_t inner,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);

int metal_dispatch_geglu(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef destinationRef,
    MetalBufferRef gateRef,
    MetalBufferRef upRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);

int metal_dispatch_glu(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef destinationRef,
    MetalBufferRef gateRef,
    MetalBufferRef upRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);

int metal_dispatch_reglu(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef destinationRef,
    MetalBufferRef gateRef,
    MetalBufferRef upRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);

int metal_dispatch_siglu(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef destinationRef,
    MetalBufferRef gateRef,
    MetalBufferRef upRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);

int metal_dispatch_seglu(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef destinationRef,
    MetalBufferRef gateRef,
    MetalBufferRef upRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);

int metal_dispatch_linglu(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef destinationRef,
    MetalBufferRef gateRef,
    MetalBufferRef upRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);

int metal_dispatch_geglu_tanh(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef destinationRef,
    MetalBufferRef gateRef,
    MetalBufferRef upRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);

int metal_dispatch_geglu_packed(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef destinationRef,
    MetalBufferRef packedRef,
    uint32_t inner,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);

int metal_dispatch_glu_packed(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef destinationRef,
    MetalBufferRef packedRef,
    uint32_t inner,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);

int metal_dispatch_reglu_packed(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef destinationRef,
    MetalBufferRef packedRef,
    uint32_t inner,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);

int metal_dispatch_siglu_packed(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef destinationRef,
    MetalBufferRef packedRef,
    uint32_t inner,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);

int metal_dispatch_seglu_packed(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef destinationRef,
    MetalBufferRef packedRef,
    uint32_t inner,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);

int metal_dispatch_linglu_packed(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef destinationRef,
    MetalBufferRef packedRef,
    uint32_t inner,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);

int metal_dispatch_geglu_tanh_packed(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef destinationRef,
    MetalBufferRef packedRef,
    uint32_t inner,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
