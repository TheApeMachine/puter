#ifndef PUTER_DEVICE_CUDA_NORMALIZATION_NORMALIZATION_CUH
#define PUTER_DEVICE_CUDA_NORMALIZATION_NORMALIZATION_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

static constexpr unsigned int normalizationThreadCountCUDA = 256u;
static constexpr float normalizationEpsilonCUDA = 1.0e-5f;

__device__ __forceinline__ float norm_bf16_to_float(unsigned short value) {
    unsigned int bits = static_cast<unsigned int>(value) << 16u;
    return __uint_as_float(bits);
}

__device__ __forceinline__ unsigned short norm_float_to_bf16(float value) {
    return static_cast<unsigned short>(__float_as_uint(value) >> 16u);
}

__device__ __forceinline__ float norm_load_f32(const float* input, unsigned int index) {
    return input[index];
}

__device__ __forceinline__ void norm_store_f32(float* output, unsigned int index, float value) {
    output[index] = value;
}

__device__ __forceinline__ float norm_load_f16(const __half* input, unsigned int index) {
    return __half2float(input[index]);
}

__device__ __forceinline__ void norm_store_f16(__half* output, unsigned int index, float value) {
    output[index] = __float2half(value);
}

__device__ __forceinline__ float norm_load_bf16(const unsigned short* input, unsigned int index) {
    return norm_bf16_to_float(input[index]);
}

__device__ __forceinline__ void norm_store_bf16(unsigned short* output, unsigned int index, float value) {
    output[index] = norm_float_to_bf16(value);
}

__device__ __forceinline__ float norm_kahan_partial_sum_f32(
    const float* input,
    unsigned int baseOffset,
    unsigned int elementCount,
    unsigned int threadIndex
) {
    float localSum = 0.0f;
    float localCompensation = 0.0f;

    for (unsigned int index = threadIndex; index < elementCount; index += normalizationThreadCountCUDA) {
        float value = input[baseOffset + index] - localCompensation;
        float nextSum = localSum + value;
        localCompensation = (nextSum - localSum) - value;
        localSum = nextSum;
    }

    return localSum;
}

__device__ __forceinline__ float norm_tree_reduce256(float* reduction) {
    for (unsigned int stride = normalizationThreadCountCUDA / 2u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }

        __syncthreads();
    }

    return reduction[0];
}

__device__ __forceinline__ float norm_reduce_sum_f32(
    const float* input,
    unsigned int baseOffset,
    unsigned int elementCount
) {
    __shared__ float reduction[normalizationThreadCountCUDA];
    reduction[threadIdx.x] = norm_kahan_partial_sum_f32(input, baseOffset, elementCount, threadIdx.x);
    __syncthreads();
    return norm_tree_reduce256(reduction);
}

#define NORM_LOAD_STORE_F16 \
    __device__ __forceinline__ float norm_load(const __half* input, unsigned int index) { \
        return norm_load_f16(input, index); \
    } \
    __device__ __forceinline__ void norm_store(__half* output, unsigned int index, float value) { \
        norm_store_f16(output, index, value); \
    }

#define NORM_LOAD_STORE_BF16 \
    __device__ __forceinline__ float norm_load(const unsigned short* input, unsigned int index) { \
        return norm_load_bf16(input, index); \
    } \
    __device__ __forceinline__ void norm_store(unsigned short* output, unsigned int index, float value) { \
        norm_store_bf16(output, index, value); \
    }

#endif
