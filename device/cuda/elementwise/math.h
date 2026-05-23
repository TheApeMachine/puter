#ifndef PUTER_DEVICE_CUDA_ELEMENTWISE_MATH_H
#define PUTER_DEVICE_CUDA_ELEMENTWISE_MATH_H

#include "elementwise.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum CUDAUnaryMathOp {
    CUDAUnaryMathAbs = 0,
    CUDAUnaryMathNeg = 1,
    CUDAUnaryMathSqrt = 2,
    CUDAUnaryMathReLU = 3,
} CUDAUnaryMathOp;

int cuda_dispatch_unary_math(
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
