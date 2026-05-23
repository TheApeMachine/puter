#ifndef PUTER_DEVICE_CUDA_VSA_DISPATCH_H
#define PUTER_DEVICE_CUDA_VSA_DISPATCH_H

#include "vsa.h"

#ifdef __cplusplus
extern "C" {
#endif

void cuda_vsa_register_module_source(const char* source);

int cuda_dispatch_vsa_binary(
    CUDADeviceRef contextRef,
    int operation,
    int elementDType,
    CUDABufferRef leftRef,
    CUDABufferRef rightRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

int cuda_dispatch_vsa_unary(
    CUDADeviceRef contextRef,
    int operation,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
