#ifndef PUTER_DEVICE_CUDA_PHYSICS_SPECTRAL_H
#define PUTER_DEVICE_CUDA_PHYSICS_SPECTRAL_H

#include "physics.h"

#ifdef __cplusplus
extern "C" {
#endif

int cuda_dispatch_fft1d(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef realInRef,
    CUDABufferRef imagInRef,
    CUDABufferRef realOutRef,
    CUDABufferRef imagOutRef,
    CUDABufferRef twiddleRealRef,
    CUDABufferRef twiddleImagRef,
    uint32_t count,
    int inverse,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
