#ifndef PUTER_DEVICE_CUDA_LAYERNORM_LAYERNORM_CUH
#define PUTER_DEVICE_CUDA_LAYERNORM_LAYERNORM_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

static constexpr unsigned int normalizationThreadCount = 256U;
static constexpr float layerNormEpsilonCUDA = 1.0e-5f;

__device__ __forceinline__ float bf16_to_float_norm(unsigned short value) {
    unsigned int bits = static_cast<unsigned int>(value) << 16U;
    return __uint_as_float(bits);
}

__device__ __forceinline__ unsigned short float_to_bf16_norm(float value) {
    return static_cast<unsigned short>(__float_as_uint(value) >> 16U);
}

__device__ __forceinline__ float kahan_partial_sum(
    const float* input,
    unsigned int baseOffset,
    unsigned int elementCount,
    unsigned int threadIndex
) {
    float localSum = 0.0f;
    float localCompensation = 0.0f;

    for (unsigned int index = threadIndex; index < elementCount; index += normalizationThreadCount) {
        float value = input[baseOffset + index] - localCompensation;
        float nextSum = localSum + value;
        localCompensation = (nextSum - localSum) - value;
        localSum = nextSum;
    }

    return localSum;
}

__device__ __forceinline__ float tree_reduce256(float* reduction) {
    for (unsigned int stride = normalizationThreadCount / 2U; stride > 0U; stride >>= 1U) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }

        __syncthreads();
    }

    return reduction[0];
}

__device__ __forceinline__ float reduce_sum_cuda(
    const float* input,
    unsigned int baseOffset,
    unsigned int elementCount
) {
    __shared__ float reduction[normalizationThreadCount];
    reduction[threadIdx.x] = kahan_partial_sum(input, baseOffset, elementCount, threadIdx.x);
    __syncthreads();
    return tree_reduce256(reduction);
}

__device__ __forceinline__ float kahan_partial_variance(
    const float* input,
    unsigned int baseOffset,
    unsigned int elementCount,
    float mean,
    unsigned int threadIndex
) {
    float localVariance = 0.0f;
    float localCompensation = 0.0f;

    for (unsigned int index = threadIndex; index < elementCount; index += normalizationThreadCount) {
        float delta = input[baseOffset + index] - mean;
        float value = delta * delta - localCompensation;
        float nextVariance = localVariance + value;
        localCompensation = (nextVariance - localVariance) - value;
        localVariance = nextVariance;
    }

    return localVariance;
}

__device__ __forceinline__ float plain_partial_variance(
    const float* input,
    unsigned int baseOffset,
    unsigned int elementCount,
    float mean,
    unsigned int threadIndex
) {
    float localVariance = 0.0f;

    for (unsigned int index = threadIndex; index < elementCount; index += normalizationThreadCount) {
        float delta = input[baseOffset + index] - mean;
        localVariance += delta * delta;
    }

    return localVariance;
}

#endif
