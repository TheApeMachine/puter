#include "elementwise.cuh"

#define ELEMENTWISE_UNARY_KERNEL_F32(name, expr) \
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

#define ELEMENTWISE_UNARY_KERNEL_F16(name, expr) \
extern "C" __global__ void name##_float16( \
    const __half* input, \
    __half* output, \
    unsigned int count \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (index >= count) { \
        return; \
    } \
    float value = __half2float(input[index]); \
    output[index] = __float2half((expr)); \
}

#define ELEMENTWISE_UNARY_KERNEL_BF16(name, expr) \
extern "C" __global__ void name##_bfloat16( \
    const unsigned short* input, \
    unsigned short* output, \
    unsigned int count \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (index >= count) { \
        return; \
    } \
    float value = elementwise_bf16_to_float(input[index]); \
    output[index] = elementwise_float_to_bf16((expr)); \
}

#define ELEMENTWISE_UNARY_ALL(name, expr) \
    ELEMENTWISE_UNARY_KERNEL_F32(name, expr) \
    ELEMENTWISE_UNARY_KERNEL_F16(name, expr) \
    ELEMENTWISE_UNARY_KERNEL_BF16(name, expr)

ELEMENTWISE_UNARY_ALL(abs, fabsf(value))
ELEMENTWISE_UNARY_ALL(neg, -value)
ELEMENTWISE_UNARY_ALL(sqrt, sqrtf(value))
ELEMENTWISE_UNARY_ALL(relu, value > 0.0f ? value : 0.0f)
