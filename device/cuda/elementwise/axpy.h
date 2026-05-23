#ifndef PUTER_DEVICE_CUDA_ELEMENTWISE_AXPY_H
#define PUTER_DEVICE_CUDA_ELEMENTWISE_AXPY_H

#include "elementwise.h"

#ifdef __cplusplus
extern "C" {
#endif

int cuda_dispatch_axpy(
    CUDADeviceRef contextRef,
    int elementDType,
    CUDABufferRef yRef,
    CUDABufferRef xRef,
    float alpha,
    uint32_t count,
    uint64_t completionToken,
    CUDAStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
