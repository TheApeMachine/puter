#include "activation.cuh"
#include "softmax_reduce.cuh"

#include <stdint.h>

__device__ __forceinline__ float softmax_tree_reduce256(float* reduction) {
    for (unsigned int stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }

        __syncthreads();
    }

    return reduction[0];
}

__device__ __forceinline__ float softmax_tree_reduce256_max(float* reduction) {
    for (unsigned int stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] = fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
        }

        __syncthreads();
    }

    return reduction[0];
}

__device__ __forceinline__ __half softmax_tree_reduce256_h(__half* reduction) {
    for (unsigned int stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] = __hadd(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
        }

        __syncthreads();
    }

    return reduction[0];
}

__device__ __forceinline__ __half softmax_tree_reduce256_max_h(__half* reduction) {
    for (unsigned int stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] = __hmax(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
        }

        __syncthreads();
    }

    return reduction[0];
}

__device__ __forceinline__ __nv_bfloat16 softmax_tree_reduce256_bf16(__nv_bfloat16* reduction) {
    for (unsigned int stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] = __hadd(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
        }

        __syncthreads();
    }

    return reduction[0];
}

__device__ __forceinline__ __nv_bfloat16 softmax_tree_reduce256_max_bf16(__nv_bfloat16* reduction) {
    for (unsigned int stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] = __hmax(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
        }

        __syncthreads();
    }

    return reduction[0];
}

extern "C" __global__ void softmax_float32(
    const float* input,
    float* output,
    unsigned int cols
) {
    __shared__ float reductionScratch[32];
    unsigned int row = blockIdx.x;
    unsigned int threadIndex = threadIdx.x;
    unsigned int rowOffset = row * cols;
    float localMax = -CUDART_INF_F;

    for (unsigned int col = threadIndex; col < cols; col += blockDim.x) {
        localMax = fmaxf(localMax, input[rowOffset + col]);
    }

    float rowMax = softmax_block_reduce_max(localMax, reductionScratch);
    float localSum = 0.0f;

    for (unsigned int col = threadIndex; col < cols; col += blockDim.x) {
        localSum += expf(input[rowOffset + col] - rowMax);
    }

    float rowSum = softmax_block_reduce_sum(localSum, reductionScratch);

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

    __half rowMax = softmax_tree_reduce256_max_h(rowMaxScratch);
    __half localSum = activation_zero_h();

    for (unsigned int col = threadIndex; col < cols; col += blockDim.x) {
        localSum = __hadd(localSum, hexp(__hsub(input[rowOffset + col], rowMax)));
    }

    rowSumScratch[threadIndex] = localSum;
    __syncthreads();

    __half rowSum = softmax_tree_reduce256_h(rowSumScratch);

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

    __nv_bfloat16 rowMax = softmax_tree_reduce256_max_bf16(rowMaxScratch);
    __nv_bfloat16 localSum = activation_zero_bf16();

    for (unsigned int col = threadIndex; col < cols; col += blockDim.x) {
        localSum = __hadd(localSum, activation_bf16_exp(__hsub(input[rowOffset + col], rowMax)));
    }

    rowSumScratch[threadIndex] = localSum;
    __syncthreads();

    __nv_bfloat16 rowSum = softmax_tree_reduce256_bf16(rowSumScratch);

    for (unsigned int col = threadIndex; col < cols; col += blockDim.x) {
        __nv_bfloat16 numerator = activation_bf16_exp(__hsub(input[rowOffset + col], rowMax));
        output[rowOffset + col] = __hdiv(numerator, rowSum);
    }
}
