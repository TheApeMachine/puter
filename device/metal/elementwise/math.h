#ifndef PUTER_DEVICE_METAL_ELEMENTWISE_MATH_H
#define PUTER_DEVICE_METAL_ELEMENTWISE_MATH_H

#include "elementwise.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum MetalUnaryMathOp {
    MetalUnaryMathAbs = 0,
    MetalUnaryMathNeg = 1,
    MetalUnaryMathSqrt = 2,
    MetalUnaryMathReLU = 3,
} MetalUnaryMathOp;

int metal_dispatch_unary_math(
    MetalDeviceRef contextRef,
    int operation,
    int elementDType,
    MetalBufferRef inputRef,
    MetalBufferRef outRef,
    uint32_t count,
    uint64_t completionToken,
    MetalStatus* status
);

#ifdef __cplusplus
}
#endif

#endif
