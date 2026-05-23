#ifndef PUTER_DEVICE_CUDA_DEQUANT_DISPATCH_H
#define PUTER_DEVICE_CUDA_DEQUANT_DISPATCH_H

#include "dequant.h"

#ifdef __cplusplus
extern "C" {
#endif

void cuda_dequant_register_module_source(const char* source);

int cuda_dispatch_dequantization(
    CUDADeviceRef contextRef,
    int operation,
    int elementDType,
    CUDABufferRef inputRef,
    CUDABufferRef outRef,
    float scale,
    int zeroPoint,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
