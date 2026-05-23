#ifndef PUTER_DEVICE_CUDA_QUANT_QUANT_CUH
#define PUTER_DEVICE_CUDA_QUANT_QUANT_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ int8_t quant_lane(float value, float invScale, int zeroPoint) {
    float scaled = roundf(value * invScale + static_cast<float>(zeroPoint));
    scaled = fminf(fmaxf(scaled, -128.0f), 127.0f);
    return static_cast<int8_t>(scaled);
}

#define QUANT_INT8_KERNEL_F32(name) \
extern "C" __global__ void name##_float32( \
    const float* inputRaw, \
    int8_t* output, \
    float invScale, \
    int zeroPoint, \
    unsigned int count \
) { \
    const float4* inputVector = reinterpret_cast<const float4*>(inputRaw); \
    unsigned int vectorIndex = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int base = vectorIndex * 4u; \
    if (base + 3u < count) { \
        float4 value = inputVector[vectorIndex]; \
        output[base] = quant_lane(value.x, invScale, zeroPoint); \
        output[base + 1u] = quant_lane(value.y, invScale, zeroPoint); \
        output[base + 2u] = quant_lane(value.z, invScale, zeroPoint); \
        output[base + 3u] = quant_lane(value.w, invScale, zeroPoint); \
        return; \
    } \
    for (unsigned int offset = 0u; offset < 4u; offset++) { \
        unsigned int index = base + offset; \
        if (index < count) { \
            output[index] = quant_lane(inputRaw[index], invScale, zeroPoint); \
        } \
    } \
}

#define QUANT_INT8_KERNEL_F16(name) \
extern "C" __global__ void name##_float16( \
    const __half* input, \
    int8_t* output, \
    float invScale, \
    int zeroPoint, \
    unsigned int count \
) { \
    unsigned int pairIndex = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int base = pairIndex * 2u; \
    if (base + 1u < count) { \
        output[base] = quant_lane(__half2float(input[base]), invScale, zeroPoint); \
        output[base + 1u] = quant_lane(__half2float(input[base + 1u]), invScale, zeroPoint); \
        return; \
    } \
    if (base < count) { \
        output[base] = quant_lane(__half2float(input[base]), invScale, zeroPoint); \
    } \
}

#define QUANT_INT8_KERNEL_BF16(name) \
extern "C" __global__ void name##_bfloat16( \
    const __nv_bfloat16* input, \
    int8_t* output, \
    float invScale, \
    int zeroPoint, \
    unsigned int count \
) { \
    unsigned int pairIndex = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int base = pairIndex * 2u; \
    if (base + 1u < count) { \
        output[base] = quant_lane(__bfloat162float(input[base]), invScale, zeroPoint); \
        output[base + 1u] = quant_lane(__bfloat162float(input[base + 1u]), invScale, zeroPoint); \
        return; \
    } \
    if (base < count) { \
        output[base] = quant_lane(__bfloat162float(input[base]), invScale, zeroPoint); \
    } \
}

QUANT_INT8_KERNEL_F32(int8_quant)
QUANT_INT8_KERNEL_F16(int8_quant)
QUANT_INT8_KERNEL_BF16(int8_quant)

#endif
