#include "normalization.cuh"

extern "C" __global__ void batchnorm_eval_float32(
    const float* input,
    const float* scale,
    const float* bias,
    const float* mean,
    const float* variance,
    float* output,
    unsigned int channels,
    unsigned int spatial
) {
    unsigned int row = blockIdx.x;
    unsigned int channel = row % channels;
    unsigned int rowOffset = row * spatial;
    float invStdDev = rsqrtf(variance[channel] + normalizationEpsilonCUDA);
    float channelMean = mean[channel];
    float channelScale = scale[channel];
    float channelBias = bias[channel];

    for (unsigned int offset = threadIdx.x; offset < spatial; offset += normalizationThreadCountCUDA) {
        float normalized = (input[rowOffset + offset] - channelMean) * invStdDev;
        output[rowOffset + offset] = normalized * channelScale + channelBias;
    }
}

#define BATCHNORM_EVAL_KERNEL_F16(name) \
extern "C" __global__ void name##_float16( \
    const __half* input, \
    const __half* scale, \
    const __half* bias, \
    const __half* mean, \
    const __half* variance, \
    __half* output, \
    unsigned int channels, \
    unsigned int spatial \
) { \
    NORM_LOAD_STORE_F16 \
    unsigned int row = blockIdx.x; \
    unsigned int channel = row % channels; \
    unsigned int rowOffset = row * spatial; \
    float invStdDev = rsqrtf(norm_load(variance, channel) + normalizationEpsilonCUDA); \
    float channelMean = norm_load(mean, channel); \
    float channelScale = norm_load(scale, channel); \
    float channelBias = norm_load(bias, channel); \
    for (unsigned int offset = threadIdx.x; offset < spatial; offset += normalizationThreadCountCUDA) { \
        float normalized = (norm_load(input, rowOffset + offset) - channelMean) * invStdDev; \
        norm_store(output, rowOffset + offset, normalized * channelScale + channelBias); \
    } \
}

#define BATCHNORM_EVAL_KERNEL_BF16(name) \
extern "C" __global__ void name##_bfloat16( \
    const unsigned short* input, \
    const unsigned short* scale, \
    const unsigned short* bias, \
    const unsigned short* mean, \
    const unsigned short* variance, \
    unsigned short* output, \
    unsigned int channels, \
    unsigned int spatial \
) { \
    NORM_LOAD_STORE_BF16 \
    unsigned int row = blockIdx.x; \
    unsigned int channel = row % channels; \
    unsigned int rowOffset = row * spatial; \
    float invStdDev = rsqrtf(norm_load(variance, channel) + normalizationEpsilonCUDA); \
    float channelMean = norm_load(mean, channel); \
    float channelScale = norm_load(scale, channel); \
    float channelBias = norm_load(bias, channel); \
    for (unsigned int offset = threadIdx.x; offset < spatial; offset += normalizationThreadCountCUDA) { \
        float normalized = (norm_load(input, rowOffset + offset) - channelMean) * invStdDev; \
        norm_store(output, rowOffset + offset, normalized * channelScale + channelBias); \
    } \
}

BATCHNORM_EVAL_KERNEL_F16(batchnorm_eval)
BATCHNORM_EVAL_KERNEL_BF16(batchnorm_eval)
