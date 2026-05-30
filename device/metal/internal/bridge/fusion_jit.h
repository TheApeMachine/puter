#ifndef PUTER_DEVICE_METAL_INTERNAL_BRIDGE_FUSION_JIT_H
#define PUTER_DEVICE_METAL_INTERNAL_BRIDGE_FUSION_JIT_H

#include "core.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* MetalFusionProgramRef;

MetalFusionProgramRef metal_fusion_program_compile(
    MetalDeviceRef contextRef,
    const char* source,
    const char* kernelName,
    MetalStatus* status
);

void metal_fusion_program_release(MetalFusionProgramRef programRef);

int metal_fusion_program_dispatch(
    MetalDeviceRef contextRef,
    MetalFusionProgramRef programRef,
    MetalBufferRef* inputRefs,
    MetalBufferRef outputRef,
    int inputCount,
    uint32_t count,
    MetalStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
