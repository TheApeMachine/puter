#include "normalization.cuh"

extern "C" __global__ void groupnorm_float32(
    const float* input,
    const float* scale,
    const float* bias,
    float* output,
    unsigned int channels,
    unsigned int spatial,
    unsigned int groups
) {
    __shared__ float reduction[normalizationThreadCountCUDA];
    unsigned int row = blockIdx.x;
    unsigned int groupIndex = row % groups;
    unsigned int batchIndex = row / groups;
    unsigned int channelsPerGroup = channels / groups;
    unsigned int channelStart = groupIndex * channelsPerGroup;
    unsigned int groupSize = channelsPerGroup * spatial;
    unsigned int groupOffset = (batchIndex * channels + channelStart) * spatial;

    float mean = norm_reduce_sum_f32(input, groupOffset, groupSize) / static_cast<float>(groupSize);

    float localVariance = 0.0f;

    for (unsigned int offset = threadIdx.x; offset < groupSize; offset += normalizationThreadCountCUDA) {
        float delta = input[groupOffset + offset] - mean;
        localVariance += delta * delta;
    }

    reduction[threadIdx.x] = localVariance;
    __syncthreads();

    float varianceSum = norm_tree_reduce256(reduction);
    float invStdDev = rsqrtf(varianceSum / static_cast<float>(groupSize) + normalizationEpsilonCUDA);

    for (unsigned int offset = threadIdx.x; offset < groupSize; offset += normalizationThreadCountCUDA) {
        unsigned int channel = channelStart + offset / spatial;
        float normalized = (input[groupOffset + offset] - mean) * invStdDev;
        output[groupOffset + offset] = normalized * scale[channel] + bias[channel];
    }
}

#define GROUPNORM_KERNEL_F16(name) \
extern "C" __global__ void name##_float16( \
    const __half* input, \
    const __half* scale, \
    const __half* bias, \
    __half* output, \
    unsigned int channels, \
    unsigned int spatial, \
    unsigned int groups \
) { \
    NORM_LOAD_STORE_F16 \
    __shared__ float reduction[normalizationThreadCountCUDA]; \
    unsigned int row = blockIdx.x; \
    unsigned int groupIndex = row % groups; \
    unsigned int batchIndex = row / groups; \
    unsigned int channelsPerGroup = channels / groups; \
    unsigned int channelStart = groupIndex * channelsPerGroup; \
    unsigned int groupSize = channelsPerGroup * spatial; \
    unsigned int groupOffset = (batchIndex * channels + channelStart) * spatial; \
    float localSum = 0.0f; \
    for (unsigned int offset = threadIdx.x; offset < groupSize; offset += normalizationThreadCountCUDA) { \
        localSum += norm_load(input, groupOffset + offset); \
    } \
    reduction[threadIdx.x] = localSum; \
    __syncthreads(); \
    float mean = norm_tree_reduce256(reduction) / static_cast<float>(groupSize); \
    float localVariance = 0.0f; \
    for (unsigned int offset = threadIdx.x; offset < groupSize; offset += normalizationThreadCountCUDA) { \
        float delta = norm_load(input, groupOffset + offset) - mean; \
        localVariance += delta * delta; \
    } \
    reduction[threadIdx.x] = localVariance; \
    __syncthreads(); \
    float invStdDev = rsqrtf(norm_tree_reduce256(reduction) / static_cast<float>(groupSize) + normalizationEpsilonCUDA); \
    for (unsigned int offset = threadIdx.x; offset < groupSize; offset += normalizationThreadCountCUDA) { \
        unsigned int channel = channelStart + offset / spatial; \
        float normalized = (norm_load(input, groupOffset + offset) - mean) * invStdDev; \
        norm_store(output, groupOffset + offset, normalized * norm_load(scale, channel) + norm_load(bias, channel)); \
    } \
}

#define GROUPNORM_KERNEL_BF16(name) \
extern "C" __global__ void name##_bfloat16( \
    const unsigned short* input, \
    const unsigned short* scale, \
    const unsigned short* bias, \
    unsigned short* output, \
    unsigned int channels, \
    unsigned int spatial, \
    unsigned int groups \
) { \
    NORM_LOAD_STORE_BF16 \
    __shared__ float reduction[normalizationThreadCountCUDA]; \
    unsigned int row = blockIdx.x; \
    unsigned int groupIndex = row % groups; \
    unsigned int batchIndex = row / groups; \
    unsigned int channelsPerGroup = channels / groups; \
    unsigned int channelStart = groupIndex * channelsPerGroup; \
    unsigned int groupSize = channelsPerGroup * spatial; \
    unsigned int groupOffset = (batchIndex * channels + channelStart) * spatial; \
    float localSum = 0.0f; \
    for (unsigned int offset = threadIdx.x; offset < groupSize; offset += normalizationThreadCountCUDA) { \
        localSum += norm_load(input, groupOffset + offset); \
    } \
    reduction[threadIdx.x] = localSum; \
    __syncthreads(); \
    float mean = norm_tree_reduce256(reduction) / static_cast<float>(groupSize); \
    float localVariance = 0.0f; \
    for (unsigned int offset = threadIdx.x; offset < groupSize; offset += normalizationThreadCountCUDA) { \
        float delta = norm_load(input, groupOffset + offset) - mean; \
        localVariance += delta * delta; \
    } \
    reduction[threadIdx.x] = localVariance; \
    __syncthreads(); \
    float invStdDev = rsqrtf(norm_tree_reduce256(reduction) / static_cast<float>(groupSize) + normalizationEpsilonCUDA); \
    for (unsigned int offset = threadIdx.x; offset < groupSize; offset += normalizationThreadCountCUDA) { \
        unsigned int channel = channelStart + offset / spatial; \
        float normalized = (norm_load(input, groupOffset + offset) - mean) * invStdDev; \
        norm_store(output, groupOffset + offset, normalized * norm_load(scale, channel) + norm_load(bias, channel)); \
    } \
}

GROUPNORM_KERNEL_F16(groupnorm)
GROUPNORM_KERNEL_BF16(groupnorm)
