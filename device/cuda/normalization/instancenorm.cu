#include "normalization.cuh"

extern "C" __global__ void instancenorm_float32(
    const float* input,
    const float* scale,
    const float* bias,
    float* output,
    unsigned int channels,
    unsigned int spatial
) {
    __shared__ float reduction[normalizationThreadCountCUDA];
    unsigned int row = blockIdx.x;
    unsigned int channel = row % channels;
    unsigned int rowOffset = row * spatial;

    float mean = norm_reduce_sum_f32(input, rowOffset, spatial) / static_cast<float>(spatial);
    float localVariance = 0.0f;

    for (unsigned int offset = threadIdx.x; offset < spatial; offset += normalizationThreadCountCUDA) {
        float delta = input[rowOffset + offset] - mean;
        localVariance += delta * delta;
    }

    reduction[threadIdx.x] = localVariance;
    __syncthreads();

    float invStdDev = rsqrtf(norm_tree_reduce256(reduction) / static_cast<float>(spatial) + normalizationEpsilonCUDA);

    for (unsigned int offset = threadIdx.x; offset < spatial; offset += normalizationThreadCountCUDA) {
        float normalized = (input[rowOffset + offset] - mean) * invStdDev;
        output[rowOffset + offset] = normalized * scale[channel] + bias[channel];
    }
}

#define INSTANCENORM_KERNEL_F16(name) \
extern "C" __global__ void name##_float16( \
    const __half* input, \
    const __half* scale, \
    const __half* bias, \
    __half* output, \
    unsigned int channels, \
    unsigned int spatial \
) { \
    NORM_LOAD_STORE_F16 \
    __shared__ float reduction[normalizationThreadCountCUDA]; \
    unsigned int row = blockIdx.x; \
    unsigned int channel = row % channels; \
    unsigned int rowOffset = row * spatial; \
    float localSum = 0.0f; \
    for (unsigned int offset = threadIdx.x; offset < spatial; offset += normalizationThreadCountCUDA) { \
        localSum += norm_load(input, rowOffset + offset); \
    } \
    reduction[threadIdx.x] = localSum; \
    __syncthreads(); \
    float mean = norm_tree_reduce256(reduction) / static_cast<float>(spatial); \
    float localVariance = 0.0f; \
    for (unsigned int offset = threadIdx.x; offset < spatial; offset += normalizationThreadCountCUDA) { \
        float delta = norm_load(input, rowOffset + offset) - mean; \
        localVariance += delta * delta; \
    } \
    reduction[threadIdx.x] = localVariance; \
    __syncthreads(); \
    float invStdDev = rsqrtf(norm_tree_reduce256(reduction) / static_cast<float>(spatial) + normalizationEpsilonCUDA); \
    for (unsigned int offset = threadIdx.x; offset < spatial; offset += normalizationThreadCountCUDA) { \
        float normalized = (norm_load(input, rowOffset + offset) - mean) * invStdDev; \
        norm_store(output, rowOffset + offset, normalized * norm_load(scale, channel) + norm_load(bias, channel)); \
    } \
}

#define INSTANCENORM_KERNEL_BF16(name) \
extern "C" __global__ void name##_bfloat16( \
    const unsigned short* input, \
    const unsigned short* scale, \
    const unsigned short* bias, \
    unsigned short* output, \
    unsigned int channels, \
    unsigned int spatial \
) { \
    NORM_LOAD_STORE_BF16 \
    __shared__ float reduction[normalizationThreadCountCUDA]; \
    unsigned int row = blockIdx.x; \
    unsigned int channel = row % channels; \
    unsigned int rowOffset = row * spatial; \
    float localSum = 0.0f; \
    for (unsigned int offset = threadIdx.x; offset < spatial; offset += normalizationThreadCountCUDA) { \
        localSum += norm_load(input, rowOffset + offset); \
    } \
    reduction[threadIdx.x] = localSum; \
    __syncthreads(); \
    float mean = norm_tree_reduce256(reduction) / static_cast<float>(spatial); \
    float localVariance = 0.0f; \
    for (unsigned int offset = threadIdx.x; offset < spatial; offset += normalizationThreadCountCUDA) { \
        float delta = norm_load(input, rowOffset + offset) - mean; \
        localVariance += delta * delta; \
    } \
    reduction[threadIdx.x] = localVariance; \
    __syncthreads(); \
    float invStdDev = rsqrtf(norm_tree_reduce256(reduction) / static_cast<float>(spatial) + normalizationEpsilonCUDA); \
    for (unsigned int offset = threadIdx.x; offset < spatial; offset += normalizationThreadCountCUDA) { \
        float normalized = (norm_load(input, rowOffset + offset) - mean) * invStdDev; \
        norm_store(output, rowOffset + offset, normalized * norm_load(scale, channel) + norm_load(bias, channel)); \
    } \
}

INSTANCENORM_KERNEL_F16(instancenorm)
INSTANCENORM_KERNEL_BF16(instancenorm)
