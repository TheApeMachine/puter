#ifndef PUTER_DEVICE_CUDA_SAMPLING_DISPATCH_H
#define PUTER_DEVICE_CUDA_SAMPLING_DISPATCH_H

#include "sampling.h"

#ifdef __cplusplus
extern "C" {
#endif

void cuda_sampling_register_module_source(const char* source);

int cuda_dispatch_sampling(
    CUDADeviceRef contextRef,
    int operation,
    int elementDType,
    CUDABufferRef logitsRef,
    CUDABufferRef scoresRef,
    CUDABufferRef indicesRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint32_t paddedCount,
    float target,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
