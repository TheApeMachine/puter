#ifndef PUTER_DEVICE_CUDA_LOSSES_DISPATCH_H
#define PUTER_DEVICE_CUDA_LOSSES_DISPATCH_H

#include "losses.h"

#ifdef __cplusplus
extern "C" {
#endif

void cuda_losses_register_module_source(const char* source);

int cuda_dispatch_pair_loss(
    CUDADeviceRef contextRef,
    int operation,
    int elementDType,
    CUDABufferRef predictionsRef,
    CUDABufferRef targetsRef,
    CUDABufferRef scratchRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint32_t partialCount,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_cross_entropy_loss(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef logitsRef,
    CUDABufferRef targetsRef,
    CUDABufferRef scratchRef,
    CUDABufferRef outRef,
    CUDABufferRef errorFlagRef,
    uint32_t batch,
    uint32_t classes,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
