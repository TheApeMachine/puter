#ifndef PUTER_DEVICE_CUDA_DEQUANT_DEQUANT_CUH
#define PUTER_DEVICE_CUDA_DEQUANT_DEQUANT_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ float dequant_int8_lane(int8_t value, float scale, int zeroPoint) {
    return (static_cast<float>(static_cast<int>(value) - zeroPoint)) * scale;
}

static __device__ __forceinline__ int dequant_int4_lane(int8_t packedByte, unsigned int nibbleIndex, int zeroPoint) {
    int value = (nibbleIndex == 0u) ? (static_cast<int>(packedByte) & 0x0F) : ((static_cast<int>(packedByte) >> 4) & 0x0F);

    if (value >= 8) {
        value -= 16;
    }

    return value - zeroPoint;
}

#define DEQUANT_INT8_KERNEL_F32(name) \
extern "C" __global__ void name##_float32( \
    float* destination, \
    const int8_t* source, \
    float scale, \
    int zeroPoint, \
    unsigned int count \
) { \
    const float4* destinationVector = reinterpret_cast<float4*>(destination); \
    unsigned int vectorIndex = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int base = vectorIndex * 4u; \
    if (base + 3u < count) { \
        float4 result; \
        result.x = dequant_int8_lane(source[base], scale, zeroPoint); \
        result.y = dequant_int8_lane(source[base + 1u], scale, zeroPoint); \
        result.z = dequant_int8_lane(source[base + 2u], scale, zeroPoint); \
        result.w = dequant_int8_lane(source[base + 3u], scale, zeroPoint); \
        destinationVector[vectorIndex] = result; \
        return; \
    } \
    for (unsigned int offset = 0u; offset < 4u; offset++) { \
        unsigned int index = base + offset; \
        if (index < count) { \
            destination[index] = dequant_int8_lane(source[index], scale, zeroPoint); \
        } \
    } \
}

#define DEQUANT_INT8_KERNEL_F16(name) \
extern "C" __global__ void name##_float16( \
    __half* destination, \
    const int8_t* source, \
    float scale, \
    int zeroPoint, \
    unsigned int count \
) { \
    unsigned int pairIndex = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int base = pairIndex * 2u; \
    if (base + 1u < count) { \
        destination[base] = __float2half(dequant_int8_lane(source[base], scale, zeroPoint)); \
        destination[base + 1u] = __float2half(dequant_int8_lane(source[base + 1u], scale, zeroPoint)); \
        return; \
    } \
    if (base < count) { \
        destination[base] = __float2half(dequant_int8_lane(source[base], scale, zeroPoint)); \
    } \
}

#define DEQUANT_INT8_KERNEL_BF16(name) \
extern "C" __global__ void name##_bfloat16( \
    __nv_bfloat16* destination, \
    const int8_t* source, \
    float scale, \
    int zeroPoint, \
    unsigned int count \
) { \
    unsigned int pairIndex = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int base = pairIndex * 2u; \
    if (base + 1u < count) { \
        destination[base] = __float2bfloat16(dequant_int8_lane(source[base], scale, zeroPoint)); \
        destination[base + 1u] = __float2bfloat16(dequant_int8_lane(source[base + 1u], scale, zeroPoint)); \
        return; \
    } \
    if (base < count) { \
        destination[base] = __float2bfloat16(dequant_int8_lane(source[base], scale, zeroPoint)); \
    } \
}

#define DEQUANT_INT4_KERNEL_F32(name) \
extern "C" __global__ void name##_float32( \
    float* destination, \
    const int8_t* source, \
    float scale, \
    int zeroPoint, \
    unsigned int pairCount \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (index >= pairCount) { \
        return; \
    } \
    unsigned int byteIndex = index / 2u; \
    unsigned int nibble = index & 1u; \
    destination[index] = static_cast<float>(dequant_int4_lane(source[byteIndex], nibble, zeroPoint)) * scale; \
}

#define DEQUANT_INT4_KERNEL_F16(name) \
extern "C" __global__ void name##_float16( \
    __half* destination, \
    const int8_t* source, \
    float scale, \
    int zeroPoint, \
    unsigned int pairCount \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (index >= pairCount) { \
        return; \
    } \
    unsigned int byteIndex = index / 2u; \
    unsigned int nibble = index & 1u; \
    float value = static_cast<float>(dequant_int4_lane(source[byteIndex], nibble, zeroPoint)) * scale; \
    destination[index] = __float2half(value); \
}

#define DEQUANT_INT4_KERNEL_BF16(name) \
extern "C" __global__ void name##_bfloat16( \
    __nv_bfloat16* destination, \
    const int8_t* source, \
    float scale, \
    int zeroPoint, \
    unsigned int pairCount \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (index >= pairCount) { \
        return; \
    } \
    unsigned int byteIndex = index / 2u; \
    unsigned int nibble = index & 1u; \
    float value = static_cast<float>(dequant_int4_lane(source[byteIndex], nibble, zeroPoint)) * scale; \
    destination[index] = __float2bfloat16(value); \
}

DEQUANT_INT8_KERNEL_F32(int8_dequant)
DEQUANT_INT8_KERNEL_F16(int8_dequant)
DEQUANT_INT8_KERNEL_BF16(int8_dequant)

DEQUANT_INT4_KERNEL_F32(int4_dequant)
DEQUANT_INT4_KERNEL_F16(int4_dequant)
DEQUANT_INT4_KERNEL_BF16(int4_dequant)

#endif
