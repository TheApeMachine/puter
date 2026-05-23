#include "elementwise.cuh"

#define ELEMENTWISE_UNARY_KERNEL(name, expr) \
extern "C" __global__ void name##_float32( \
    const float* input, \
    float* output, \
    unsigned int count \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (index >= count) { \
        return; \
    } \
    float value = input[index]; \
    output[index] = (expr); \
}

ELEMENTWISE_UNARY_KERNEL(abs, fabsf(value))
ELEMENTWISE_UNARY_KERNEL(neg, -value)
ELEMENTWISE_UNARY_KERNEL(sqrt, sqrtf(value))
ELEMENTWISE_UNARY_KERNEL(relu, value > 0.0f ? value : 0.0f)

#define ELEMENTWISE_BINARY_KERNEL(name, expr) \
extern "C" __global__ void name##_float32( \
    const float* left, \
    const float* right, \
    float* output, \
    unsigned int count \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (index >= count) { \
        return; \
    } \
    float leftValue = left[index]; \
    float rightValue = right[index]; \
    output[index] = (expr); \
}

ELEMENTWISE_BINARY_KERNEL(add, leftValue + rightValue)
ELEMENTWISE_BINARY_KERNEL(sub, leftValue - rightValue)
ELEMENTWISE_BINARY_KERNEL(mul, leftValue * rightValue)
ELEMENTWISE_BINARY_KERNEL(div, leftValue / rightValue)
ELEMENTWISE_BINARY_KERNEL(max, fmaxf(leftValue, rightValue))
ELEMENTWISE_BINARY_KERNEL(min, fminf(leftValue, rightValue))
