#ifndef PUTER_DEVICE_METAL_EMBEDDING_TIMESTEP_H
#define PUTER_DEVICE_METAL_EMBEDDING_TIMESTEP_H

#include "embedding.h"

#ifdef __cplusplus
extern "C" {
#endif

int metal_dispatch_timestep_embedding(
    MetalDeviceRef contextRef,
    int elementDType,
    MetalBufferRef timestepsRef,
    MetalBufferRef outRef,
    float maxPeriod,
    float downscaleFreqShift,
    float timestepDivisor,
    int flipSinToCos,
    uint32_t count,
    uint32_t dim,
    uint64_t completionToken,
    MetalStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
