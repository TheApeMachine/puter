#ifndef PUTER_DEVICE_CUDA_QUANT_DISPATCH_H
#define PUTER_DEVICE_CUDA_QUANT_DISPATCH_H

#include "quant.h"

#ifdef __cplusplus
extern "C" {
#endif

void cuda_quant_register_module_source(const char* source);

int cuda_dispatch_quantization(
    CUDADeviceRef contextRef,
    int operation,
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
