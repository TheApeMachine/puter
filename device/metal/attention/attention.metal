#ifndef PUTER_DEVICE_METAL_ATTENTION_ATTENTION_METAL
#define PUTER_DEVICE_METAL_ATTENTION_ATTENTION_METAL

#include <metal_stdlib>

using namespace metal;

static inline float transformer_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort transformer_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

struct Float32TransformerStorage {
    static float load(device const float* values, uint index) {
        return values[index];
    }

    static void store(device float* values, uint index, float value) {
        values[index] = value;
    }
};

struct Float16TransformerStorage {
    static float load(device const half* values, uint index) {
        return float(values[index]);
    }

    static void store(device half* values, uint index, float value) {
        values[index] = half(value);
    }
};

struct BFloat16TransformerStorage {
    static float load(device const ushort* values, uint index) {
        return transformer_bf16_to_float(values[index]);
    }

    static void store(device ushort* values, uint index, float value) {
        values[index] = transformer_float_to_bf16(value);
    }
};

__attribute__((unused))
static inline void set_transformer_error(device atomic_uint* errorFlag) {
    atomic_store_explicit(errorFlag, 1u, memory_order_relaxed);
}

template <typename Storage, typename Scalar>
static inline void embedding_lookup_kernel(
    device const Scalar* table,
    device const int* indices,
    device Scalar* out,
    device atomic_uint* errorFlag,
    constant uint& vocab,
    constant uint& hidden,
    constant uint& indexCount,
    uint outputIndex
) {
    uint total = indexCount * hidden;

    if (outputIndex >= total) {
        return;
    }

    uint tokenOffset = outputIndex / hidden;
    uint hiddenOffset = outputIndex - tokenOffset * hidden;
    int tokenID = indices[tokenOffset];

    if (tokenID < 0 || uint(tokenID) >= vocab) {
        set_transformer_error(errorFlag);
        return;
    }

    out[outputIndex] = table[uint(tokenID) * hidden + hiddenOffset];
}

template <typename Storage, typename Scalar>
static inline void embedding_bag_kernel(
    device const Scalar* table,
    device const int* indices,
    device const int* offsets,
    device Scalar* out,
    device atomic_uint* errorFlag,
    constant uint& vocab,
    constant uint& hidden,
    constant uint& indexCount,
    constant uint& bagCount,
    uint outputIndex
) {
    uint total = bagCount * hidden;

    if (outputIndex >= total) {
        return;
    }

    uint bagIndex = outputIndex / hidden;
    uint hiddenOffset = outputIndex - bagIndex * hidden;
    int start = offsets[bagIndex];
    int end = bagIndex + 1 < bagCount ? offsets[bagIndex + 1] : int(indexCount);

    if (start < 0 || end < start || uint(end) > indexCount) {
        set_transformer_error(errorFlag);
        return;
    }

    float accumulator = 0.0f;

    for (int indexCursor = start; indexCursor < end; indexCursor++) {
        int tokenID = indices[indexCursor];

        if (tokenID < 0 || uint(tokenID) >= vocab) {
            set_transformer_error(errorFlag);
            return;
        }

        accumulator += Storage::load(table, uint(tokenID) * hidden + hiddenOffset);
    }

    Storage::store(out, outputIndex, accumulator);
}

template <typename Storage, typename Scalar>
static inline void attention_scores_tiled(
    device const Scalar* query,
    device const Scalar* key,
    device float* scores,
    threadgroup float* queryTile,
    threadgroup float* keyTile,
    constant uint& seqQ,
    constant uint& seqK,
    constant uint& depth,
    uint2 localPosition,
    uint2 groupPosition
) {
    uint row = groupPosition.y * 16 + localPosition.y;
    uint col = groupPosition.x * 16 + localPosition.x;
    uint localOffset = localPosition.y * 16 + localPosition.x;
    float accumulator = 0.0f;

    for (uint tileStart = 0; tileStart < depth; tileStart += 16) {
        uint queryDepth = tileStart + localPosition.x;
        uint keyDepth = tileStart + localPosition.y;

        queryTile[localOffset] =
            row < seqQ && queryDepth < depth ? Storage::load(query, row * depth + queryDepth) : 0.0f;
        keyTile[localOffset] =
            col < seqK && keyDepth < depth ? Storage::load(key, col * depth + keyDepth) : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint tileIndex = 0; tileIndex < 16; tileIndex++) {
            accumulator += queryTile[localPosition.y * 16 + tileIndex] *
                keyTile[tileIndex * 16 + localPosition.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < seqQ && col < seqK) {
        scores[row * seqK + col] = accumulator * rsqrt(float(depth));
    }
}

__attribute__((unused))
static inline void attention_softmax_row(
    device float* scores,
    threadgroup float* reduction,
    constant uint& seqK,
    uint row,
    uint threadIndex
) {
    uint rowOffset = row * seqK;
    float localMax = -3.4028234663852886e38f;

    for (uint col = threadIndex; col < seqK; col += 256) {
        localMax = max(localMax, scores[rowOffset + col]);
    }

    reduction[threadIndex] = localMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            reduction[threadIndex] = max(reduction[threadIndex], reduction[threadIndex + stride]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float maximum = reduction[0];
    float localSum = 0.0f;

    for (uint col = threadIndex; col < seqK; col += 256) {
        localSum += exp(scores[rowOffset + col] - maximum);
    }

    reduction[threadIndex] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            reduction[threadIndex] += reduction[threadIndex + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float sum = reduction[0];

    for (uint col = threadIndex; col < seqK; col += 256) {
        scores[rowOffset + col] = sum == 0.0f ? 0.0f : exp(scores[rowOffset + col] - maximum) / sum;
    }
}

template <typename Storage, typename Scalar>
static inline void attention_weighted_tiled(
    device const float* scores,
    device const Scalar* value,
    device Scalar* out,
    threadgroup float* scoreTile,
    threadgroup float* valueTile,
    constant uint& seqQ,
    constant uint& seqK,
    constant uint& valueDim,
    uint2 localPosition,
    uint2 groupPosition
) {
    uint row = groupPosition.y * 16 + localPosition.y;
    uint col = groupPosition.x * 16 + localPosition.x;
    uint localOffset = localPosition.y * 16 + localPosition.x;
    float accumulator = 0.0f;

    for (uint tileStart = 0; tileStart < seqK; tileStart += 16) {
        uint scoreCol = tileStart + localPosition.x;
        uint valueRow = tileStart + localPosition.y;

        scoreTile[localOffset] =
            row < seqQ && scoreCol < seqK ? scores[row * seqK + scoreCol] : 0.0f;
        valueTile[localOffset] =
            valueRow < seqK && col < valueDim ? Storage::load(value, valueRow * valueDim + col) : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint tileIndex = 0; tileIndex < 16; tileIndex++) {
            accumulator += scoreTile[localPosition.y * 16 + tileIndex] *
                valueTile[tileIndex * 16 + localPosition.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < seqQ && col < valueDim) {
        Storage::store(out, row * valueDim + col, accumulator);
    }
}

template <typename Storage, typename Scalar>
static inline void flash_attention_online(
    device const Scalar* query,
    device const Scalar* key,
    device const Scalar* value,
    device Scalar* out,
    constant uint& seqQ,
    constant uint& seqK,
    constant uint& depth,
    constant uint& valueDim,
    threadgroup float* reduction,
    uint2 groupPosition,
    uint2 localPosition
) {
    if (groupPosition.x >= seqQ) {
        return;
    }

    uint threadIndex = localPosition.x;
    uint row = groupPosition.x;
    uint valueColumn = groupPosition.y * 64 + threadIndex;
    float maxScore = -3.4028234663852886e38f;
    float normalizer = 0.0f;
    float accumulator = 0.0f;
    float scale = rsqrt(float(depth));

    for (uint keyIndex = 0; keyIndex < seqK; keyIndex++) {
        float localDot = 0.0f;

        for (uint depthIndex = threadIndex; depthIndex < depth; depthIndex += 256) {
            localDot += Storage::load(query, row * depth + depthIndex) *
                Storage::load(key, keyIndex * depth + depthIndex);
        }

        reduction[threadIndex] = localDot;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 128; stride > 0; stride >>= 1) {
            if (threadIndex < stride) {
                reduction[threadIndex] += reduction[threadIndex + stride];
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        float dot = reduction[0];
        float score = dot * scale;
        float oldMax = maxScore;
        maxScore = max(maxScore, score);
        float alpha = exp(oldMax - maxScore);
        float shifted = exp(score - maxScore);
        normalizer = normalizer * alpha + shifted;

        if (threadIndex < 64 && valueColumn < valueDim) {
            accumulator = accumulator * alpha +
                shifted * Storage::load(value, keyIndex * valueDim + valueColumn);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (threadIndex < 64 && valueColumn < valueDim) {
        float outputValue = normalizer == 0.0f ? 0.0f : accumulator / normalizer;
        Storage::store(out, row * valueDim + valueColumn, outputValue);
    }
}

__attribute__((unused))
static inline bool attention_variant_keeps_key(
    uint row,
    uint keyIndex,
    uint seqQ,
    uint seqK,
    uint causal,
    uint windowSize
) {
    uint absoluteRow = row + seqK - seqQ;

    if (causal != 0 && keyIndex > absoluteRow) {
        return false;
    }

    if (windowSize != 0 && row >= keyIndex && row - keyIndex >= windowSize) {
        return false;
    }

    return true;
}

template <typename Storage, typename Scalar>
static inline void multi_head_attention_online(
    device const Scalar* query,
    device const Scalar* key,
    device const Scalar* value,
    device Scalar* out,
    constant uint& seqQ,
    constant uint& seqK,
    constant uint& numHeads,
    constant uint& kvHeads,
    constant uint& headDim,
    constant uint& windowSize,
    constant uint& causal,
    threadgroup float* reduction,
    uint3 groupPosition,
    uint3 localPosition
) {
    if (groupPosition.x >= seqQ || groupPosition.y >= numHeads) {
        return;
    }

    uint threadIndex = localPosition.x;
    uint dimIndex = groupPosition.z * 64 + threadIndex;

    uint row = groupPosition.x;
    uint headIndex = groupPosition.y;
    uint headsPerKVHead = numHeads / kvHeads;
    uint kvHeadIndex = headIndex / headsPerKVHead;
    uint queryStride = numHeads * headDim;
    uint kvStride = kvHeads * headDim;
    uint queryHeadOffset = headIndex * headDim;
    uint kvHeadOffset = kvHeadIndex * headDim;
    float maxScore = -3.4028234663852886e38f;
    float normalizer = 0.0f;
    float accumulator = 0.0f;
    float scale = rsqrt(float(headDim));

    for (uint keyIndex = 0; keyIndex < seqK; keyIndex++) {
        bool keepKey = attention_variant_keeps_key(row, keyIndex, seqQ, seqK, causal, windowSize);

        if (threadIndex == 0) {
            float dot = 0.0f;

            if (keepKey) {
                for (uint depthIndex = 0; depthIndex < headDim; depthIndex++) {
                    dot += Storage::load(
                    query, row * queryStride + queryHeadOffset + depthIndex
                    ) * Storage::load(
                    key, keyIndex * kvStride + kvHeadOffset + depthIndex
                    );
                }
            }

            reduction[0] = dot;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (keepKey) {
            float score = reduction[0] * scale;
            float oldMax = maxScore;
            maxScore = max(maxScore, score);
            float alpha = exp(oldMax - maxScore);
            float shifted = exp(score - maxScore);
            normalizer = normalizer * alpha + shifted;

            if (threadIndex < 64 && dimIndex < headDim) {
                accumulator = accumulator * alpha + shifted * Storage::load(
                    value, keyIndex * kvStride + kvHeadOffset + dimIndex
                );
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (threadIndex < 64 && dimIndex < headDim) {
        float outputValue = normalizer == 0.0f ? 0.0f : accumulator / normalizer;
        Storage::store(out, row * queryStride + queryHeadOffset + dimIndex, outputValue);
    }
}

__attribute__((unused))
static inline float llama3_scaled_inv_freq(
    float invFreq,
    uint originalContext,
    float factor,
    float lowFreqFactor,
    float highFreqFactor
) {
    float wavelen = (2.0f * M_PI_F) / invFreq;
    float lowFreqWavelen = float(originalContext) / lowFreqFactor;
    float highFreqWavelen = float(originalContext) / highFreqFactor;

    if (wavelen > lowFreqWavelen) {
        return invFreq / factor;
    }

    if (wavelen < highFreqWavelen) {
        return invFreq;
    }

    float smooth = (float(originalContext) / wavelen - lowFreqFactor) /
        (highFreqFactor - lowFreqFactor);

    return (1.0f - smooth) * (invFreq / factor) + smooth * invFreq;
}

template <typename Storage, typename Scalar>
static inline void rope_kernel(
    device const Scalar* input,
    device Scalar* out,
    constant uint& seqLen,
    constant uint& numHeads,
    constant uint& headDim,
    constant uint& pairCount,
    constant float& ropeTheta,
    constant float& ropeFactor,
    constant float& lowFreqFactor,
    constant float& highFreqFactor,
    constant uint& originalContext,
    constant uint& halfMode,
    constant uint& positionOffset,
    uint index
) {
    if (index >= pairCount) {
        return;
    }

    uint halfDim = headDim / 2;
    uint pairIndex = index % halfDim;
    uint headIndex = (index / halfDim) % numHeads;
    uint seqIndex = index / (halfDim * numHeads);
    uint headOffset = (seqIndex * numHeads + headIndex) * headDim;
    uint evenIndex = halfMode != 0 ? headOffset + pairIndex : headOffset + pairIndex * 2;
    uint oddIndex = halfMode != 0 ? headOffset + halfDim + pairIndex : evenIndex + 1;
    float exponent = -2.0f * float(pairIndex) / float(headDim);
    float invFreq = pow(ropeTheta, exponent);

    if (ropeFactor > 1.0f) {
        invFreq = llama3_scaled_inv_freq(
            invFreq, originalContext, ropeFactor, lowFreqFactor, highFreqFactor
        );
    }

    float angle = float(positionOffset + seqIndex) * invFreq;
    float cosTheta = precise::cos(angle);
    float sinTheta = precise::sin(angle);
    float even = Storage::load(input, evenIndex);
    float odd = Storage::load(input, oddIndex);

    Storage::store(out, evenIndex, even * cosTheta - odd * sinTheta);
    Storage::store(out, oddIndex, even * sinTheta + odd * cosTheta);
}

template <typename Storage, typename Scalar>
static inline void multi_axis_rope_kernel(
    device const Scalar* input,
    device Scalar* out,
    constant uint& seqLen,
    constant uint& numHeads,
    constant uint& headDim,
    constant uint& pairCount,
    constant uint& latentSeqLen,
    constant uint& latentSide,
    constant float& theta,
    uint index
) {
    if (index >= pairCount) {
        return;
    }

    uint halfDim = headDim / 2;
    uint pairIndex = index % halfDim;
    uint headIndex = (index / halfDim) % numHeads;
    uint seqIndex = index / (halfDim * numHeads);
    uint inputIndex = (seqIndex * numHeads + headIndex) * headDim + pairIndex * 2;
    uint textLen = seqLen > latentSeqLen ? seqLen - latentSeqLen : 0;
    uint axisPairCount = halfDim / 4;
    uint axisIndex = axisPairCount == 0 ? 0 : pairIndex / axisPairCount;
    uint localPair = axisPairCount == 0 ? pairIndex : pairIndex - axisIndex * axisPairCount;
    uint position = 0;

    if (seqIndex < textLen) {
        if (axisIndex == 3) {
            position = seqIndex;
        }
    } else {
        uint imageIndex = seqIndex - textLen;

        if (axisIndex == 1) {
            position = imageIndex / latentSide;
        } else if (axisIndex == 2) {
            position = imageIndex % latentSide;
        }
    }

    float axisDim = float(axisPairCount * 2);
    float exponent = axisDim == 0.0f ? 0.0f : -2.0f * float(localPair) / axisDim;
    float angle = float(position) * pow(theta, exponent);
    float cosTheta = precise::cos(angle);
    float sinTheta = precise::sin(angle);
    float even = Storage::load(input, inputIndex);
    float odd = Storage::load(input, inputIndex + 1);

    Storage::store(out, inputIndex, even * cosTheta - odd * sinTheta);
    Storage::store(out, inputIndex + 1, even * sinTheta + odd * cosTheta);
}

static inline float masking_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort masking_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

__attribute__((unused))
static inline float4 masking_bf16_to_float4(ushort4 value) {
    return float4(
        masking_bf16_to_float(value.x),
        masking_bf16_to_float(value.y),
        masking_bf16_to_float(value.z),
        masking_bf16_to_float(value.w)
    );
}

__attribute__((unused))
static inline ushort4 masking_float4_to_bf16(float4 value) {
    return ushort4(
        masking_float_to_bf16(value.x),
        masking_float_to_bf16(value.y),
        masking_float_to_bf16(value.z),
        masking_float_to_bf16(value.w)
    );
}

static inline float masking_neg_inf_float32() {
    return as_type<float>(0xFF800000u);
}

__attribute__((unused))
static inline half masking_neg_inf_float16() {
    return as_type<half>(ushort(0xFC00));
}

__attribute__((unused))
static inline ushort masking_neg_inf_bfloat16() {
    return masking_float_to_bf16(masking_neg_inf_float32());
}


#define EMBEDDING_LOOKUP_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* table [[buffer(0)]], \
    device const int* indices [[buffer(1)]], \
    device scalar* out [[buffer(2)]], \
    device atomic_uint* errorFlag [[buffer(3)]], \
    constant uint& vocab [[buffer(4)]], \
    constant uint& hidden [[buffer(5)]], \
    constant uint& indexCount [[buffer(6)]], \
    uint index [[thread_position_in_grid]] \
) { \
    embedding_lookup_kernel<storage, scalar>( \
        table, indices, out, errorFlag, vocab, hidden, indexCount, index \
    ); \
}

#define EMBEDDING_BAG_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* table [[buffer(0)]], \
    device const int* indices [[buffer(1)]], \
    device const int* offsets [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    device atomic_uint* errorFlag [[buffer(4)]], \
    constant uint& vocab [[buffer(5)]], \
    constant uint& hidden [[buffer(6)]], \
    constant uint& indexCount [[buffer(7)]], \
    constant uint& bagCount [[buffer(8)]], \
    uint index [[thread_position_in_grid]] \
) { \
    embedding_bag_kernel<storage, scalar>( \
        table, indices, offsets, out, errorFlag, vocab, hidden, indexCount, bagCount, index \
    ); \
}

#define ATTENTION_SCORES_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* query [[buffer(0)]], \
    device const scalar* key [[buffer(1)]], \
    device float* scores [[buffer(2)]], \
    constant uint& seqQ [[buffer(3)]], \
    constant uint& seqK [[buffer(4)]], \
    constant uint& depth [[buffer(5)]], \
    uint2 localPosition [[thread_position_in_threadgroup]], \
    uint2 groupPosition [[threadgroup_position_in_grid]] \
) { \
    threadgroup float queryTile[256]; \
    threadgroup float keyTile[256]; \
    attention_scores_tiled<storage, scalar>( \
        query, key, scores, queryTile, keyTile, \
        seqQ, seqK, depth, localPosition, groupPosition \
    ); \
}

#define ATTENTION_WEIGHTED_KERNEL(name, storage, scalar) \
kernel void name( \
    device const float* scores [[buffer(0)]], \
    device const scalar* value [[buffer(1)]], \
    device scalar* out [[buffer(2)]], \
    constant uint& seqQ [[buffer(3)]], \
    constant uint& seqK [[buffer(4)]], \
    constant uint& valueDim [[buffer(5)]], \
    uint2 localPosition [[thread_position_in_threadgroup]], \
    uint2 groupPosition [[threadgroup_position_in_grid]] \
) { \
    threadgroup float scoreTile[256]; \
    threadgroup float valueTile[256]; \
    attention_weighted_tiled<storage, scalar>( \
        scores, value, out, scoreTile, valueTile, \
        seqQ, seqK, valueDim, localPosition, groupPosition \
    ); \
}

#define FLASH_ATTENTION_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* query [[buffer(0)]], \
    device const scalar* key [[buffer(1)]], \
    device const scalar* value [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    constant uint& seqQ [[buffer(4)]], \
    constant uint& seqK [[buffer(5)]], \
    constant uint& depth [[buffer(6)]], \
    constant uint& valueDim [[buffer(7)]], \
    uint2 groupPosition [[threadgroup_position_in_grid]], \
    uint2 localPosition [[thread_position_in_threadgroup]] \
) { \
    threadgroup float reduction[256]; \
    flash_attention_online<storage, scalar>( \
        query, key, value, out, seqQ, seqK, depth, valueDim, \
        reduction, groupPosition, localPosition \
    ); \
}

#define MULTI_HEAD_ATTENTION_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* query [[buffer(0)]], \
    device const scalar* key [[buffer(1)]], \
    device const scalar* value [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    constant uint& seqQ [[buffer(4)]], \
    constant uint& seqK [[buffer(5)]], \
    constant uint& numHeads [[buffer(6)]], \
    constant uint& kvHeads [[buffer(7)]], \
    constant uint& headDim [[buffer(8)]], \
    constant uint& windowSize [[buffer(9)]], \
    constant uint& causal [[buffer(10)]], \
    uint3 groupPosition [[threadgroup_position_in_grid]], \
    uint3 localPosition [[thread_position_in_threadgroup]] \
) { \
    threadgroup float reduction[256]; \
    multi_head_attention_online<storage, scalar>( \
        query, key, value, out, seqQ, seqK, numHeads, kvHeads, headDim, \
        windowSize, causal, reduction, groupPosition, localPosition \
    ); \
}

#define ROPE_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device scalar* out [[buffer(1)]], \
    constant uint& seqLen [[buffer(2)]], \
    constant uint& numHeads [[buffer(3)]], \
    constant uint& headDim [[buffer(4)]], \
    constant uint& pairCount [[buffer(5)]], \
    constant float& ropeTheta [[buffer(6)]], \
    constant float& ropeFactor [[buffer(7)]], \
    constant float& lowFreqFactor [[buffer(8)]], \
    constant float& highFreqFactor [[buffer(9)]], \
    constant uint& originalContext [[buffer(10)]], \
    constant uint& halfMode [[buffer(11)]], \
    constant uint& positionOffset [[buffer(12)]], \
    uint index [[thread_position_in_grid]] \
) { \
    rope_kernel<storage, scalar>( \
        input, out, seqLen, numHeads, headDim, pairCount, ropeTheta, \
        ropeFactor, lowFreqFactor, highFreqFactor, originalContext, halfMode, positionOffset, index \
    ); \
}

#define MULTI_AXIS_ROPE_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device scalar* out [[buffer(1)]], \
    constant uint& seqLen [[buffer(2)]], \
    constant uint& numHeads [[buffer(3)]], \
    constant uint& headDim [[buffer(4)]], \
    constant uint& pairCount [[buffer(5)]], \
    constant uint& latentSeqLen [[buffer(6)]], \
    constant uint& latentSide [[buffer(7)]], \
    constant float& theta [[buffer(8)]], \
    uint index [[thread_position_in_grid]] \
) { \
    multi_axis_rope_kernel<storage, scalar>( \
        input, out, seqLen, numHeads, headDim, pairCount, latentSeqLen, latentSide, theta, index \
    ); \
}

#endif
