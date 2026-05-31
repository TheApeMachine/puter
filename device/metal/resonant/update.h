#ifndef PUTER_DEVICE_METAL_RESONANT_UPDATE_H
#define PUTER_DEVICE_METAL_RESONANT_UPDATE_H

#include "resonant.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t n;
    uint32_t D;
    uint32_t H;
    float inv_D;
    float scale;
    float damping;
    uint32_t zero_diag;
} MetalResonantUpdateParams;

int metal_dispatch_resonant_update_forward(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef xRef,
    MetalBufferRef yRef,
    MetalBufferRef vrRef,
    MetalBufferRef viRef,
    MetalBufferRef diagRef,
    MetalBufferRef xOutRef,
    MetalBufferRef yOutRef,
    MetalBufferRef aOutRef,
    MetalBufferRef bOutRef,
    MetalBufferRef invROutRef,
    const MetalResonantUpdateParams* params,
    uint64_t completionToken,
    MetalStatus* status
);

int metal_dispatch_resonant_update_backward(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef gradXOutRef,
    MetalBufferRef gradYOutRef,
    MetalBufferRef xRef,
    MetalBufferRef yRef,
    MetalBufferRef diagRef,
    MetalBufferRef aRef,
    MetalBufferRef bRef,
    MetalBufferRef invRRef,
    MetalBufferRef gradXRef,
    MetalBufferRef gradYRef,
    MetalBufferRef gradVRRef,
    MetalBufferRef gradVIRef,
    const MetalResonantUpdateParams* params,
    uint64_t completionToken,
    MetalStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
