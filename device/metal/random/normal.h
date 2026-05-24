#ifndef PUTER_DEVICE_METAL_RANDOM_NORMAL_H
#define PUTER_DEVICE_METAL_RANDOM_NORMAL_H

#include "random.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
metal_dispatch_random_normal launches the random_normal_float32 kernel
on the Metal device referenced by contextRef, writing `count` standard-
normal float32s into outRef. The kernel computes Philox-4×32-10 from
(seedLo, seedHi, ctrLo, ctrHi), then Box-Muller from the four uniforms
per Philox block.
*/
int metal_dispatch_random_normal(
    MetalDeviceRef contextRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint32_t seedLo,
    uint32_t seedHi,
    uint32_t ctrLo,
    uint32_t ctrHi,
    uint64_t completionToken,
    MetalStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
