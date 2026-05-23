#include "elementwise.cuh"

#define ELEMENTWISE_BINARY_KERNEL_F32(name, expr) \
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

#define ELEMENTWISE_BINARY_KERNEL_F16(name, expr) \
extern "C" __global__ void name##_float16( \
    const __half* left, \
    const __half* right, \
    __half* output, \
    unsigned int count \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (index >= count) { \
        return; \
    } \
    float leftValue = __half2float(left[index]); \
    float rightValue = __half2float(right[index]); \
    output[index] = __float2half((expr)); \
}

#define ELEMENTWISE_BINARY_KERNEL_BF16(name, expr) \
extern "C" __global__ void name##_bfloat16( \
    const unsigned short* left, \
    const unsigned short* right, \
    unsigned short* output, \
    unsigned int count \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (index >= count) { \
        return; \
    } \
    float leftValue = elementwise_bf16_to_float(left[index]); \
    float rightValue = elementwise_bf16_to_float(right[index]); \
    output[index] = elementwise_float_to_bf16((expr)); \
}

#define ELEMENTWISE_BINARY_ALL(name, expr) \
    ELEMENTWISE_BINARY_KERNEL_F32(name, expr) \
    ELEMENTWISE_BINARY_KERNEL_F16(name, expr) \
    ELEMENTWISE_BINARY_KERNEL_BF16(name, expr)

ELEMENTWISE_BINARY_ALL(add, leftValue + rightValue)
ELEMENTWISE_BINARY_ALL(sub, leftValue - rightValue)
ELEMENTWISE_BINARY_ALL(mul, leftValue * rightValue)
ELEMENTWISE_BINARY_ALL(div, leftValue / rightValue)
ELEMENTWISE_BINARY_ALL(max, fmaxf(leftValue, rightValue))
ELEMENTWISE_BINARY_ALL(min, fminf(leftValue, rightValue))
