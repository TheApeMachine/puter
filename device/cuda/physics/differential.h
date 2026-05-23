#ifndef PUTER_DEVICE_CUDA_PHYSICS_DIFFERENTIAL_H
#define PUTER_DEVICE_CUDA_PHYSICS_DIFFERENTIAL_H

#include "physics.h"

#ifdef __cplusplus
extern "C" {
#endif

int cuda_dispatch_laplacian(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef spacingRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint32_t rank,
    uint32_t dim0,
    uint32_t dim1,
    uint32_t dim2,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_laplacian4(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef spacingRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_grad1d(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef spacingRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_divergence1d(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef spacingRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_quantum_potential(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef densityRef,
    CUDABufferRef spacingRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_bohmian_velocity(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef phaseRef,
    CUDABufferRef spacingRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_madelung_continuity(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef densityRef,
    CUDABufferRef velocityRef,
    CUDABufferRef spacingRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
