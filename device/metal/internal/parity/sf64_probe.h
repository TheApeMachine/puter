#ifndef PUTER_DEVICE_METAL_INTERNAL_PARITY_SF64_PROBE_H
#define PUTER_DEVICE_METAL_INTERNAL_PARITY_SF64_PROBE_H

#include "../bridge/core.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    MetalSF64ProbeOutputWords = 10,
    MetalSF64ProbeInputFloats = 4,
};

/*
metal_dispatch_sf64_transcendental_probe runs sf64_transcendental_probe for
caseCount rows. inputsRef holds caseCount * MetalSF64ProbeInputFloats float32
values (uniformFirst, uniformSecond, geluValue, reserved). sqrtInputsRef
holds caseCount uint64 f64 bit patterns for sqrt / invStdDev. outputsRef
receives caseCount * MetalSF64ProbeOutputWords uint64 lanes.
*/
int metal_dispatch_sf64_transcendental_probe(
    MetalDeviceRef contextRef,
    MetalBufferRef inputsRef,
    MetalBufferRef sqrtInputsRef,
    MetalBufferRef outputsRef,
    uint32_t caseCount,
    uint64_t completionToken,
    MetalStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
