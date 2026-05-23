#ifndef PUTER_DEVICE_CUDA_ELEMENTWISE_ARITHMETIC_H
#define PUTER_DEVICE_CUDA_ELEMENTWISE_ARITHMETIC_H

#include "elementwise.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum CUDABinaryFloat32Op {
    CUDABinaryFloat32Add = 0,
    CUDABinaryFloat32Sub = 1,
    CUDABinaryFloat32Mul = 2,
    CUDABinaryFloat32Div = 3,
    CUDABinaryFloat32Max = 4,
    CUDABinaryFloat32Min = 5,
    CUDABinaryFloat32Eq = 6,
    CUDABinaryFloat32Ne = 7,
    CUDABinaryFloat32Lt = 8,
    CUDABinaryFloat32Le = 9,
    CUDABinaryFloat32Gt = 10,
    CUDABinaryFloat32Ge = 11,
    CUDABinaryFloat32Pow = 12,
    CUDABinaryFloat32Atan2 = 13,
    CUDABinaryFloat32Mod = 14,
} CUDABinaryFloat32Op;

int cuda_dispatch_binary_elementwise(
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

#ifdef __cplusplus
}
#endif

#endif
