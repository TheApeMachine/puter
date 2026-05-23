#include "activation.cuh"

#include <stdint.h>

extern "C" __global__ void softmax_float32(
    const float* input,
    float* output,
    unsigned int cols
) {
    extern __shared__ float sharedScratch[];
    float* rowMaxScratch = sharedScratch;
    float* rowSumScratch = sharedScratch + blockDim.x;

    unsigned int row = blockIdx.x;
    unsigned int threadIndex = threadIdx.x;
    unsigned int rowOffset = row * cols;
    float localMax = -CUDART_INF_F;

    for (unsigned int col = threadIndex; col < cols; col += blockDim.x) {
        localMax = fmaxf(localMax, input[rowOffset + col]);
    }

    rowMaxScratch[threadIndex] = localMax;
    __syncthreads();

    if (threadIndex == 0) {
        float rowMax = rowMaxScratch[0];

        for (unsigned int offset = 1; offset < blockDim.x; offset++) {
            rowMax = fmaxf(rowMax, rowMaxScratch[offset]);
        }

        rowMaxScratch[0] = rowMax;
    }

    __syncthreads();

    float rowMax = rowMaxScratch[0];
    float localSum = 0.0f;

    for (unsigned int col = threadIndex; col < cols; col += blockDim.x) {
        localSum += expf(input[rowOffset + col] - rowMax);
    }

    rowSumScratch[threadIndex] = localSum;
    __syncthreads();

    if (threadIndex == 0) {
        float rowSum = rowSumScratch[0];

        for (unsigned int offset = 1; offset < blockDim.x; offset++) {
            rowSum += rowSumScratch[offset];
        }

        rowSumScratch[0] = rowSum;
    }

    __syncthreads();

    float rowSum = rowSumScratch[0];

    for (unsigned int col = threadIndex; col < cols; col += blockDim.x) {
        output[rowOffset + col] = expf(input[rowOffset + col] - rowMax) / rowSum;
    }
}

extern "C" __global__ void softmax_float16(
    const __half* input,
    __half* output,
    unsigned int cols
) {
    extern __shared__ __half sharedScratch[];
    __half* rowMaxScratch = sharedScratch;
    __half* rowSumScratch = sharedScratch + blockDim.x;

    unsigned int row = blockIdx.x;
    unsigned int threadIndex = threadIdx.x;
    unsigned int rowOffset = row * cols;
    __half localMax = __float2half(-65504.0f);

    for (unsigned int col = threadIndex; col < cols; col += blockDim.x) {
        localMax = __hmax(localMax, input[rowOffset + col]);
    }

    rowMaxScratch[threadIndex] = localMax;
    __syncthreads();

    if (threadIndex == 0) {
        __half rowMax = rowMaxScratch[0];

        for (unsigned int offset = 1; offset < blockDim.x; offset++) {
            rowMax = __hmax(rowMax, rowMaxScratch[offset]);
        }

        rowMaxScratch[0] = rowMax;
    }

    __syncthreads();

    __half rowMax = rowMaxScratch[0];
    __half localSum = activation_zero_h();

    for (unsigned int col = threadIndex; col < cols; col += blockDim.x) {
        localSum = __hadd(localSum, hexp(__hsub(input[rowOffset + col], rowMax)));
    }

    rowSumScratch[threadIndex] = localSum;
    __syncthreads();

    if (threadIndex == 0) {
        __half rowSum = rowSumScratch[0];

        for (unsigned int offset = 1; offset < blockDim.x; offset++) {
            rowSum = __hadd(rowSum, rowSumScratch[offset]);
        }

        rowSumScratch[0] = rowSum;
    }

    __syncthreads();

    __half rowSum = rowSumScratch[0];

    for (unsigned int col = threadIndex; col < cols; col += blockDim.x) {
        __half numerator = hexp(__hsub(input[rowOffset + col], rowMax));
        output[rowOffset + col] = __hdiv(numerator, rowSum);
    }
}

extern "C" __global__ void softmax_bfloat16(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    unsigned int cols
) {
    extern __shared__ __nv_bfloat16 sharedScratch[];
    __nv_bfloat16* rowMaxScratch = sharedScratch;
    __nv_bfloat16* rowSumScratch = sharedScratch + blockDim.x;

    unsigned int row = blockIdx.x;
    unsigned int threadIndex = threadIdx.x;
    unsigned int rowOffset = row * cols;
    __nv_bfloat16 localMax = __float2bfloat16(-3.38953139e38f);

    for (unsigned int col = threadIndex; col < cols; col += blockDim.x) {
        localMax = __hmax(localMax, input[rowOffset + col]);
    }

    rowMaxScratch[threadIndex] = localMax;
    __syncthreads();

    if (threadIndex == 0) {
        __nv_bfloat16 rowMax = rowMaxScratch[0];

        for (unsigned int offset = 1; offset < blockDim.x; offset++) {
            rowMax = __hmax(rowMax, rowMaxScratch[offset]);
        }

        rowMaxScratch[0] = rowMax;
    }

    __syncthreads();

    __nv_bfloat16 rowMax = rowMaxScratch[0];
    __nv_bfloat16 localSum = activation_zero_bf16();

    for (unsigned int col = threadIndex; col < cols; col += blockDim.x) {
        localSum = __hadd(localSum, activation_bf16_exp(__hsub(input[rowOffset + col], rowMax)));
    }

    rowSumScratch[threadIndex] = localSum;
    __syncthreads();

    if (threadIndex == 0) {
        __nv_bfloat16 rowSum = rowSumScratch[0];

        for (unsigned int offset = 1; offset < blockDim.x; offset++) {
            rowSum = __hadd(rowSum, rowSumScratch[offset]);
        }

        rowSumScratch[0] = rowSum;
    }

    __syncthreads();

    __nv_bfloat16 rowSum = rowSumScratch[0];

    for (unsigned int col = threadIndex; col < cols; col += blockDim.x) {
        __nv_bfloat16 numerator = activation_bf16_exp(__hsub(input[rowOffset + col], rowMax));
        output[rowOffset + col] = __hdiv(numerator, rowSum);
    }
}
