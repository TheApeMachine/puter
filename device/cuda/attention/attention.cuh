#ifndef PUTER_DEVICE_CUDA_ATTENTION_ATTENTION_CUH
#define PUTER_DEVICE_CUDA_ATTENTION_ATTENTION_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

static __device__ __forceinline__ float attention_load_f32(const float* values, unsigned int index) {
    return values[index];
}

static __device__ __forceinline__ void attention_store_f32(float* values, unsigned int index, float value) {
    values[index] = value;
}

static __device__ __forceinline__ float attention_load_f16(const __half* values, unsigned int index) {
    return __half2float(values[index]);
}

static __device__ __forceinline__ void attention_store_f16(__half* values, unsigned int index, float value) {
    values[index] = __float2half(value);
}

static __device__ __forceinline__ float attention_load_bf16(const __nv_bfloat16* values, unsigned int index) {
    return __bfloat162float(values[index]);
}

static __device__ __forceinline__ void attention_store_bf16(__nv_bfloat16* values, unsigned int index, float value) {
    values[index] = __float2bfloat16(value);
}

template <typename LoadFn, typename StoreFn, typename Scalar>
static __device__ __forceinline__ void attention_scores_tiled(
    const Scalar* query,
    const Scalar* key,
    float* scores,
    float* queryTile,
    float* keyTile,
    unsigned int seqQ,
    unsigned int seqK,
    unsigned int depth,
    unsigned int localX,
    unsigned int localY,
    unsigned int groupX,
    unsigned int groupY,
    LoadFn loadFn,
    StoreFn storeFn
) {
    unsigned int row = groupY * 16u + localY;
    unsigned int col = groupX * 16u + localX;
    unsigned int localOffset = localY * 16u + localX;
    float accumulator = 0.0f;

    for (unsigned int tileStart = 0; tileStart < depth; tileStart += 16u) {
        unsigned int queryDepth = tileStart + localX;
        unsigned int keyDepth = tileStart + localY;

        queryTile[localOffset] =
            row < seqQ && queryDepth < depth ? loadFn(query, row * depth + queryDepth) : 0.0f;
        keyTile[localOffset] =
            col < seqK && keyDepth < depth ? loadFn(key, col * depth + keyDepth) : 0.0f;

        __syncthreads();

        for (unsigned int tileIndex = 0; tileIndex < 16u; tileIndex++) {
            accumulator += queryTile[localY * 16u + tileIndex] *
                keyTile[tileIndex * 16u + localX];
        }

        __syncthreads();
    }

    if (row < seqQ && col < seqK) {
        scores[row * seqK + col] = accumulator * rsqrtf(static_cast<float>(depth));
    }

    (void)storeFn;
}

static __device__ __forceinline__ void attention_softmax_row(
    float* scores,
    float* reduction,
    unsigned int seqK,
    unsigned int row,
    unsigned int threadIndex
) {
    unsigned int rowOffset = row * seqK;
    float localMax = -CUDART_INF_F;

    for (unsigned int col = threadIndex; col < seqK; col += 256u) {
        localMax = fmaxf(localMax, scores[rowOffset + col]);
    }

    reduction[threadIndex] = localMax;
    __syncthreads();

    for (unsigned int stride = 128u; stride > 0u; stride >>= 1u) {
        if (threadIndex < stride) {
            reduction[threadIndex] = fmaxf(reduction[threadIndex], reduction[threadIndex + stride]);
        }

        __syncthreads();
    }

    float maximum = reduction[0];
    float localSum = 0.0f;

    for (unsigned int col = threadIndex; col < seqK; col += 256u) {
        localSum += expf(scores[rowOffset + col] - maximum);
    }

    reduction[threadIndex] = localSum;
    __syncthreads();

    for (unsigned int stride = 128u; stride > 0u; stride >>= 1u) {
        if (threadIndex < stride) {
            reduction[threadIndex] += reduction[threadIndex + stride];
        }

        __syncthreads();
    }

    float sum = reduction[0];

    for (unsigned int col = threadIndex; col < seqK; col += 256u) {
        scores[rowOffset + col] = sum == 0.0f ? 0.0f : expf(scores[rowOffset + col] - maximum) / sum;
    }
}

template <typename LoadFn, typename StoreFn, typename Scalar>
static __device__ __forceinline__ void attention_weighted_tiled(
    const float* scores,
    const Scalar* value,
    Scalar* out,
    float* scoreTile,
    float* valueTile,
    unsigned int seqQ,
    unsigned int seqK,
    unsigned int valueDim,
    unsigned int localX,
    unsigned int localY,
    unsigned int groupX,
    unsigned int groupY,
    LoadFn loadFn,
    StoreFn storeFn
) {
    unsigned int row = groupY * 16u + localY;
    unsigned int col = groupX * 16u + localX;
    unsigned int localOffset = localY * 16u + localX;
    float accumulator = 0.0f;

    for (unsigned int tileStart = 0; tileStart < seqK; tileStart += 16u) {
        unsigned int scoreCol = tileStart + localX;
        unsigned int valueRow = tileStart + localY;

        scoreTile[localOffset] =
            row < seqQ && scoreCol < seqK ? scores[row * seqK + scoreCol] : 0.0f;
        valueTile[localOffset] =
            valueRow < seqK && col < valueDim ? loadFn(value, valueRow * valueDim + col) : 0.0f;

        __syncthreads();

        for (unsigned int tileIndex = 0; tileIndex < 16u; tileIndex++) {
            accumulator += scoreTile[localY * 16u + tileIndex] *
                valueTile[tileIndex * 16u + localX];
        }

        __syncthreads();
    }

    if (row < seqQ && col < valueDim) {
        storeFn(out, row * valueDim + col, accumulator);
    }
}

#define ATTENTION_SCORES_KERNEL(name, scalarType, loadFn, storeFn) \
extern "C" __global__ void name( \
    const scalarType* query, \
    const scalarType* key, \
    float* scores, \
    unsigned int seqQ, \
    unsigned int seqK, \
    unsigned int depth \
) { \
    __shared__ float queryTile[256]; \
    __shared__ float keyTile[256]; \
    attention_scores_tiled( \
        query, key, scores, queryTile, keyTile, \
        seqQ, seqK, depth, \
        threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, \
        loadFn, storeFn \
    ); \
}

#define ATTENTION_WEIGHTED_KERNEL(name, scalarType, loadFn, storeFn) \
extern "C" __global__ void name( \
    const float* scores, \
    const scalarType* value, \
    scalarType* out, \
    unsigned int seqQ, \
    unsigned int seqK, \
    unsigned int valueDim \
) { \
    __shared__ float scoreTile[256]; \
    __shared__ float valueTile[256]; \
    attention_weighted_tiled( \
        scores, value, out, scoreTile, valueTile, \
        seqQ, seqK, valueDim, \
        threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, \
        loadFn, storeFn \
    ); \
}

extern "C" __global__ void attention_softmax(
    float* scores,
    unsigned int seqK,
    unsigned int seqQ
) {
    __shared__ float reduction[256];
    unsigned int row = blockIdx.x;

    if (row >= seqQ) {
        return;
    }

    attention_softmax_row(scores, reduction, seqK, row, threadIdx.x);
}

ATTENTION_SCORES_KERNEL(attention_scores_float32, float, attention_load_f32, attention_store_f32)
ATTENTION_SCORES_KERNEL(attention_scores_float16, __half, attention_load_f16, attention_store_f16)
ATTENTION_SCORES_KERNEL(attention_scores_bfloat16, __nv_bfloat16, attention_load_bf16, attention_store_bf16)

ATTENTION_WEIGHTED_KERNEL(attention_weighted_float32, float, attention_load_f32, attention_store_f32)
ATTENTION_WEIGHTED_KERNEL(attention_weighted_float16, __half, attention_load_f16, attention_store_f16)
ATTENTION_WEIGHTED_KERNEL(attention_weighted_bfloat16, __nv_bfloat16, attention_load_bf16, attention_store_bf16)

template <typename LoadFn, typename StoreFn, typename Scalar>
static __device__ __forceinline__ void flash_attention_online(
    const Scalar* query,
    const Scalar* key,
    const Scalar* value,
    Scalar* out,
    unsigned int seqQ,
    unsigned int seqK,
    unsigned int depth,
    unsigned int valueDim,
    float* reduction,
    unsigned int groupX,
    unsigned int groupY,
    unsigned int threadIndex,
    LoadFn loadFn,
    StoreFn storeFn
) {
    if (groupX >= seqQ) {
        return;
    }

    unsigned int row = groupX;
    unsigned int valueColumn = groupY * 64u + threadIndex;
    float maxScore = -CUDART_INF_F;
    float normalizer = 0.0f;
    float accumulator = 0.0f;
    float scale = rsqrtf(static_cast<float>(depth));

    for (unsigned int keyIndex = 0; keyIndex < seqK; keyIndex++) {
        float localDot = 0.0f;

        for (unsigned int depthIndex = threadIndex; depthIndex < depth; depthIndex += 256u) {
            localDot += loadFn(query, row * depth + depthIndex) *
                loadFn(key, keyIndex * depth + depthIndex);
        }

        reduction[threadIndex] = localDot;
        __syncthreads();

        for (unsigned int stride = 128u; stride > 0u; stride >>= 1u) {
            if (threadIndex < stride) {
                reduction[threadIndex] += reduction[threadIndex + stride];
            }

            __syncthreads();
        }

        float dot = reduction[0];
        float score = dot * scale;
        float oldMax = maxScore;
        maxScore = fmaxf(maxScore, score);
        float alpha = expf(oldMax - maxScore);
        float shifted = expf(score - maxScore);
        normalizer = normalizer * alpha + shifted;

        if (threadIndex < 64u && valueColumn < valueDim) {
            accumulator = accumulator * alpha +
                shifted * loadFn(value, keyIndex * valueDim + valueColumn);
        }

        __syncthreads();
    }

    if (threadIndex < 64u && valueColumn < valueDim) {
        float outputValue = normalizer == 0.0f ? 0.0f : accumulator / normalizer;
        storeFn(out, row * valueDim + valueColumn, outputValue);
    }
}

#define FLASH_ATTENTION_KERNEL(name, scalarType, loadFn, storeFn) \
extern "C" __global__ void name( \
    const scalarType* query, \
    const scalarType* key, \
    const scalarType* value, \
    scalarType* out, \
    unsigned int seqQ, \
    unsigned int seqK, \
    unsigned int depth, \
    unsigned int valueDim \
) { \
    __shared__ float reduction[256]; \
    flash_attention_online( \
        query, key, value, out, \
        seqQ, seqK, depth, valueDim, reduction, \
        blockIdx.x, blockIdx.y, threadIdx.x, loadFn, storeFn \
    ); \
}

FLASH_ATTENTION_KERNEL(flash_attention_float32, float, attention_load_f32, attention_store_f32)
FLASH_ATTENTION_KERNEL(flash_attention_float16, __half, attention_load_f16, attention_store_f16)
FLASH_ATTENTION_KERNEL(flash_attention_bfloat16, __nv_bfloat16, attention_load_bf16, attention_store_bf16)

static __device__ __forceinline__ bool attention_variant_keeps_key(
    unsigned int row,
    unsigned int keyIndex,
    unsigned int causal,
    unsigned int windowSize
) {
    if (causal != 0u && keyIndex > row) {
        return false;
    }

    if (windowSize != 0u && row >= keyIndex && row - keyIndex >= windowSize) {
        return false;
    }

    return true;
}

template <typename LoadFn, typename StoreFn, typename Scalar>
static __device__ __forceinline__ void multi_head_attention_online(
    const Scalar* query,
    const Scalar* key,
    const Scalar* value,
    Scalar* out,
    unsigned int seqQ,
    unsigned int seqK,
    unsigned int numHeads,
    unsigned int kvHeads,
    unsigned int headDim,
    unsigned int windowSize,
    unsigned int causal,
    float* reduction,
    unsigned int groupX,
    unsigned int groupY,
    unsigned int groupZ,
    unsigned int threadIndex,
    LoadFn loadFn,
    StoreFn storeFn
) {
    if (groupX >= seqQ || groupY >= numHeads) {
        return;
    }

    unsigned int dimIndex = groupZ * 64u + threadIndex;
    unsigned int row = groupX;
    unsigned int headIndex = groupY;
    unsigned int headsPerKVHead = numHeads / kvHeads;
    unsigned int kvHeadIndex = headIndex / headsPerKVHead;
    unsigned int queryStride = numHeads * headDim;
    unsigned int kvStride = kvHeads * headDim;
    unsigned int queryHeadOffset = headIndex * headDim;
    unsigned int kvHeadOffset = kvHeadIndex * headDim;
    float maxScore = -CUDART_INF_F;
    float normalizer = 0.0f;
    float accumulator = 0.0f;
    float scale = rsqrtf(static_cast<float>(headDim));

    for (unsigned int keyIndex = 0; keyIndex < seqK; keyIndex++) {
        bool keepKey = attention_variant_keeps_key(row, keyIndex, causal, windowSize);

        if (threadIndex == 0u) {
            float dot = 0.0f;

            if (keepKey) {
                for (unsigned int depthIndex = 0; depthIndex < headDim; depthIndex++) {
                    dot += loadFn(query, row * queryStride + queryHeadOffset + depthIndex) *
                        loadFn(key, keyIndex * kvStride + kvHeadOffset + depthIndex);
                }
            }

            reduction[0] = dot;
        }

        __syncthreads();

        if (keepKey) {
            float score = reduction[0] * scale;
            float oldMax = maxScore;
            maxScore = fmaxf(maxScore, score);
            float alpha = expf(oldMax - maxScore);
            float shifted = expf(score - maxScore);
            normalizer = normalizer * alpha + shifted;

            if (threadIndex < 64u && dimIndex < headDim) {
                accumulator = accumulator * alpha +
                    shifted * loadFn(value, keyIndex * kvStride + kvHeadOffset + dimIndex);
            }
        }

        __syncthreads();
    }

    if (threadIndex < 64u && dimIndex < headDim) {
        float outputValue = normalizer == 0.0f ? 0.0f : accumulator / normalizer;
        storeFn(out, row * queryStride + queryHeadOffset + dimIndex, outputValue);
    }
}

#define MULTI_HEAD_ATTENTION_KERNEL(name, scalarType, loadFn, storeFn) \
extern "C" __global__ void name( \
    const scalarType* query, \
    const scalarType* key, \
    const scalarType* value, \
    scalarType* out, \
    unsigned int seqQ, \
    unsigned int seqK, \
    unsigned int numHeads, \
    unsigned int kvHeads, \
    unsigned int headDim, \
    unsigned int windowSize, \
    unsigned int causal \
) { \
    __shared__ float reduction[256]; \
    multi_head_attention_online( \
        query, key, value, out, \
        seqQ, seqK, numHeads, kvHeads, headDim, windowSize, causal, \
        reduction, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, loadFn, storeFn \
    ); \
}

MULTI_HEAD_ATTENTION_KERNEL(multi_head_attention_float32, float, attention_load_f32, attention_store_f32)
MULTI_HEAD_ATTENTION_KERNEL(multi_head_attention_float16, __half, attention_load_f16, attention_store_f16)
MULTI_HEAD_ATTENTION_KERNEL(multi_head_attention_bfloat16, __nv_bfloat16, attention_load_bf16, attention_store_bf16)

MULTI_HEAD_ATTENTION_KERNEL(grouped_query_attention_float32, float, attention_load_f32, attention_store_f32)
MULTI_HEAD_ATTENTION_KERNEL(grouped_query_attention_float16, __half, attention_load_f16, attention_store_f16)
MULTI_HEAD_ATTENTION_KERNEL(grouped_query_attention_bfloat16, __nv_bfloat16, attention_load_bf16, attention_store_bf16)

MULTI_HEAD_ATTENTION_KERNEL(sliding_window_attention_float32, float, attention_load_f32, attention_store_f32)
MULTI_HEAD_ATTENTION_KERNEL(sliding_window_attention_float16, __half, attention_load_f16, attention_store_f16)
MULTI_HEAD_ATTENTION_KERNEL(sliding_window_attention_bfloat16, __nv_bfloat16, attention_load_bf16, attention_store_bf16)

#endif
