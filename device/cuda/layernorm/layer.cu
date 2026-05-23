#include "layernorm.cuh"

__device__ __forceinline__ float load_half(const __half* values, unsigned int index) {
    return __half2float(values[index]);
}

__device__ __forceinline__ void store_half(__half* values, unsigned int index, float value) {
    values[index] = __float2half(value);
}

__device__ __forceinline__ float load_bf16(const unsigned short* values, unsigned int index) {
    return bf16_to_float_norm(values[index]);
}

__device__ __forceinline__ void store_bf16(unsigned short* values, unsigned int index, float value) {
    values[index] = float_to_bf16_norm(value);
}

__device__ __forceinline__ float kahan_partial_variance_half(
    const __half* input,
    unsigned int baseOffset,
    unsigned int elementCount,
    float mean,
    unsigned int threadIndex
) {
    float localVariance = 0.0f;
    float localCompensation = 0.0f;

    for (unsigned int index = threadIndex; index < elementCount; index += normalizationThreadCount) {
        float delta = load_half(input, baseOffset + index) - mean;
        float value = delta * delta - localCompensation;
        float nextVariance = localVariance + value;
        localCompensation = (nextVariance - localVariance) - value;
        localVariance = nextVariance;
    }

    return localVariance;
}

__device__ __forceinline__ float kahan_partial_variance_bf16(
    const unsigned short* input,
    unsigned int baseOffset,
    unsigned int elementCount,
    float mean,
    unsigned int threadIndex
) {
    float localVariance = 0.0f;
    float localCompensation = 0.0f;

    for (unsigned int index = threadIndex; index < elementCount; index += normalizationThreadCount) {
        float delta = load_bf16(input, baseOffset + index) - mean;
        float value = delta * delta - localCompensation;
        float nextVariance = localVariance + value;
        localCompensation = (nextVariance - localVariance) - value;
        localVariance = nextVariance;
    }

    return localVariance;
}

extern "C" __global__ void layernorm_float32(
    const float* input,
    const float* scale,
    const float* bias,
    float* output,
    unsigned int cols
) {
    unsigned int rowOffset = blockIdx.x * cols;
    float mean = reduce_sum_cuda(input, rowOffset, cols) / static_cast<float>(cols);

    __shared__ float reduction[normalizationThreadCount];
    reduction[threadIdx.x] = kahan_partial_variance(input, rowOffset, cols, mean, threadIdx.x);
    __syncthreads();

    float varianceSum = tree_reduce256(reduction);
    float invStdDev = rsqrtf(varianceSum / static_cast<float>(cols) + layerNormEpsilonCUDA);

    for (unsigned int column = threadIdx.x; column < cols; column += normalizationThreadCount) {
        float normalized = (input[rowOffset + column] - mean) * invStdDev;
        output[rowOffset + column] = normalized * scale[column] + bias[column];
    }
}

__device__ __forceinline__ float kahan_partial_sum_half(
    const __half* input,
    unsigned int baseOffset,
    unsigned int elementCount,
    unsigned int threadIndex
) {
    float localSum = 0.0f;
    float localCompensation = 0.0f;

    for (unsigned int index = threadIndex; index < elementCount; index += normalizationThreadCount) {
        float value = load_half(input, baseOffset + index) - localCompensation;
        float nextSum = localSum + value;
        localCompensation = (nextSum - localSum) - value;
        localSum = nextSum;
    }

    return localSum;
}

__device__ __forceinline__ float kahan_partial_sum_bf16(
    const unsigned short* input,
    unsigned int baseOffset,
    unsigned int elementCount,
    unsigned int threadIndex
) {
    float localSum = 0.0f;
    float localCompensation = 0.0f;

    for (unsigned int index = threadIndex; index < elementCount; index += normalizationThreadCount) {
        float value = load_bf16(input, baseOffset + index) - localCompensation;
        float nextSum = localSum + value;
        localCompensation = (nextSum - localSum) - value;
        localSum = nextSum;
    }

    return localSum;
}

extern "C" __global__ void layernorm_float16(
    const __half* input,
    const __half* scale,
    const __half* bias,
    __half* output,
    unsigned int cols
) {
    unsigned int rowOffset = blockIdx.x * cols;

    __shared__ float reduction[normalizationThreadCount];
    reduction[threadIdx.x] = kahan_partial_sum_half(input, rowOffset, cols, threadIdx.x);
    __syncthreads();

    float mean = tree_reduce256(reduction) / static_cast<float>(cols);

    reduction[threadIdx.x] = kahan_partial_variance_half(input, rowOffset, cols, mean, threadIdx.x);
    __syncthreads();

    float varianceSum = tree_reduce256(reduction);
    float invStdDev = rsqrtf(varianceSum / static_cast<float>(cols) + layerNormEpsilonCUDA);

    for (unsigned int column = threadIdx.x; column < cols; column += normalizationThreadCount) {
        float loaded = load_half(input, rowOffset + column);
        float normalized = (loaded - mean) * invStdDev;
        float result = normalized * load_half(scale, column) + load_half(bias, column);
        store_half(output, rowOffset + column, result);
    }
}

extern "C" __global__ void layernorm_bfloat16(
    const unsigned short* input,
    const unsigned short* scale,
    const unsigned short* bias,
    unsigned short* output,
    unsigned int cols
) {
    unsigned int rowOffset = blockIdx.x * cols;

    __shared__ float reduction[normalizationThreadCount];
    reduction[threadIdx.x] = kahan_partial_sum_bf16(input, rowOffset, cols, threadIdx.x);
    __syncthreads();

    float mean = tree_reduce256(reduction) / static_cast<float>(cols);

    reduction[threadIdx.x] = kahan_partial_variance_bf16(input, rowOffset, cols, mean, threadIdx.x);
    __syncthreads();

    float varianceSum = tree_reduce256(reduction);
    float invStdDev = rsqrtf(varianceSum / static_cast<float>(cols) + layerNormEpsilonCUDA);

    for (unsigned int column = threadIdx.x; column < cols; column += normalizationThreadCount) {
        float loaded = load_bf16(input, rowOffset + column);
        float normalized = (loaded - mean) * invStdDev;
        float result = normalized * load_bf16(scale, column) + load_bf16(bias, column);
        store_bf16(output, rowOffset + column, result);
    }
}

extern "C" __global__ void rmsnorm_float32(
    const float* input,
    const float* scale,
    float* output,
    unsigned int cols
) {
    unsigned int rowOffset = blockIdx.x * cols;

    __shared__ float reduction[normalizationThreadCount];
    float localSquareSum = 0.0f;
    float localCompensation = 0.0f;

    for (unsigned int column = threadIdx.x; column < cols; column += normalizationThreadCount) {
        float value = input[rowOffset + column];
        float square = value * value - localCompensation;
        float nextSum = localSquareSum + square;
        localCompensation = (nextSum - localSquareSum) - square;
        localSquareSum = nextSum;
    }

    reduction[threadIdx.x] = localSquareSum;
    __syncthreads();

    float varianceSum = tree_reduce256(reduction);
    float invRMS = rsqrtf(varianceSum / static_cast<float>(cols) + rmsNormEpsilonCUDA);

    for (unsigned int column = threadIdx.x; column < cols; column += normalizationThreadCount) {
        output[rowOffset + column] = input[rowOffset + column] * invRMS * scale[column];
    }
}

extern "C" __global__ void rmsnorm_float16(
    const __half* input,
    const __half* scale,
    __half* output,
    unsigned int cols
) {
    unsigned int rowOffset = blockIdx.x * cols;

    __shared__ float reduction[normalizationThreadCount];
    float localSquareSum = 0.0f;
    float localCompensation = 0.0f;

    for (unsigned int column = threadIdx.x; column < cols; column += normalizationThreadCount) {
        float value = load_half(input, rowOffset + column);
        float square = value * value - localCompensation;
        float nextSum = localSquareSum + square;
        localCompensation = (nextSum - localSquareSum) - square;
        localSquareSum = nextSum;
    }

    reduction[threadIdx.x] = localSquareSum;
    __syncthreads();

    float varianceSum = tree_reduce256(reduction);
    float invRMS = rsqrtf(varianceSum / static_cast<float>(cols) + rmsNormEpsilonCUDA);

    for (unsigned int column = threadIdx.x; column < cols; column += normalizationThreadCount) {
        float loaded = load_half(input, rowOffset + column);
        store_half(output, rowOffset + column, loaded * invRMS * load_half(scale, column));
    }
}

extern "C" __global__ void rmsnorm_bfloat16(
    const unsigned short* input,
    const unsigned short* scale,
    unsigned short* output,
    unsigned int cols
) {
    unsigned int rowOffset = blockIdx.x * cols;

    __shared__ float reduction[normalizationThreadCount];
    float localSquareSum = 0.0f;
    float localCompensation = 0.0f;

    for (unsigned int column = threadIdx.x; column < cols; column += normalizationThreadCount) {
        float value = load_bf16(input, rowOffset + column);
        float square = value * value - localCompensation;
        float nextSum = localSquareSum + square;
        localCompensation = (nextSum - localSquareSum) - square;
        localSquareSum = nextSum;
    }

    reduction[threadIdx.x] = localSquareSum;
    __syncthreads();

    float varianceSum = tree_reduce256(reduction);
    float invRMS = rsqrtf(varianceSum / static_cast<float>(cols) + rmsNormEpsilonCUDA);

    for (unsigned int column = threadIdx.x; column < cols; column += normalizationThreadCount) {
        float loaded = load_bf16(input, rowOffset + column);
        store_bf16(output, rowOffset + column, loaded * invRMS * load_bf16(scale, column));
    }
}
