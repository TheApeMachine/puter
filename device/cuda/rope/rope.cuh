#ifndef PUTER_DEVICE_CUDA_ROPE_ROPE_CUH
#define PUTER_DEVICE_CUDA_ROPE_ROPE_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

static __device__ __forceinline__ float rope_load_f32(const float* values, unsigned int index) {
    return values[index];
}

static __device__ __forceinline__ void rope_store_f32(float* values, unsigned int index, float value) {
    values[index] = value;
}

static __device__ __forceinline__ float rope_load_f16(const __half* values, unsigned int index) {
    return __half2float(values[index]);
}

static __device__ __forceinline__ void rope_store_f16(__half* values, unsigned int index, float value) {
    values[index] = __float2half(value);
}

static __device__ __forceinline__ float rope_load_bf16(const __nv_bfloat16* values, unsigned int index) {
    return __bfloat162float(values[index]);
}

static __device__ __forceinline__ void rope_store_bf16(__nv_bfloat16* values, unsigned int index, float value) {
    values[index] = __float2bfloat16(value);
}

static __device__ __forceinline__ float rope_llama3_scaled_inv_freq(
    float invFreq,
    unsigned int originalContext,
    float factor,
    float lowFreqFactor,
    float highFreqFactor
) {
    float wavelen = (2.0f * CUDART_PI_F) / invFreq;
    float lowFreqWavelen = static_cast<float>(originalContext) / lowFreqFactor;
    float highFreqWavelen = static_cast<float>(originalContext) / highFreqFactor;

    if (wavelen > lowFreqWavelen) {
        return invFreq / factor;
    }

    if (wavelen < highFreqWavelen) {
        return invFreq;
    }

    float smooth = (static_cast<float>(originalContext) / wavelen - lowFreqFactor) /
        (highFreqFactor - lowFreqFactor);

    return (1.0f - smooth) * (invFreq / factor) + smooth * invFreq;
}

template <typename LoadFn, typename StoreFn, typename Scalar>
static __device__ __forceinline__ void rope_apply_pair(
    const Scalar* input,
    Scalar* out,
    unsigned int evenIndex,
    unsigned int oddIndex,
    float cosTheta,
    float sinTheta,
    LoadFn loadFn,
    StoreFn storeFn
) {
    float even = loadFn(input, evenIndex);
    float odd = loadFn(input, oddIndex);
    storeFn(out, evenIndex, even * cosTheta - odd * sinTheta);
    storeFn(out, oddIndex, even * sinTheta + odd * cosTheta);
}

template <typename LoadFn, typename StoreFn, typename Scalar>
static __device__ __forceinline__ void rope_kernel_body(
    const Scalar* input,
    Scalar* out,
    unsigned int seqLen,
    unsigned int numHeads,
    unsigned int headDim,
    unsigned int pairCount,
    float ropeTheta,
    float ropeFactor,
    float lowFreqFactor,
    float highFreqFactor,
    unsigned int originalContext,
    unsigned int halfMode,
    unsigned int positionOffset,
    unsigned int index,
    LoadFn loadFn,
    StoreFn storeFn
) {
    if (index >= pairCount) {
        return;
    }

    unsigned int halfDim = headDim / 2u;
    unsigned int pairIndex = index % halfDim;
    unsigned int headIndex = (index / halfDim) % numHeads;
    unsigned int seqIndex = index / (halfDim * numHeads);
    unsigned int headOffset = (seqIndex * numHeads + headIndex) * headDim;
    unsigned int evenIndex = halfMode != 0u ? headOffset + pairIndex : headOffset + pairIndex * 2u;
    unsigned int oddIndex = halfMode != 0u ? headOffset + halfDim + pairIndex : evenIndex + 1u;
    float exponent = -2.0f * static_cast<float>(pairIndex) / static_cast<float>(headDim);
    float invFreq = powf(ropeTheta, exponent);

    if (ropeFactor > 1.0f) {
        invFreq = rope_llama3_scaled_inv_freq(
            invFreq,
            originalContext,
            ropeFactor,
            lowFreqFactor,
            highFreqFactor
        );
    }

    float angle = static_cast<float>(positionOffset + seqIndex) * invFreq;
    float cosTheta = cosf(angle);
    float sinTheta = sinf(angle);

    rope_apply_pair(input, out, evenIndex, oddIndex, cosTheta, sinTheta, loadFn, storeFn);
}

#define ROPE_KERNEL(name, scalarType, loadFn, storeFn) \
extern "C" __global__ void name( \
    const scalarType* input, \
    scalarType* out, \
    unsigned int seqLen, \
    unsigned int numHeads, \
    unsigned int headDim, \
    unsigned int pairCount, \
    float ropeTheta, \
    float ropeFactor, \
    float lowFreqFactor, \
    float highFreqFactor, \
    unsigned int originalContext, \
    unsigned int halfMode, \
    unsigned int positionOffset \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    rope_kernel_body( \
        input, out, seqLen, numHeads, headDim, pairCount, \
        ropeTheta, ropeFactor, lowFreqFactor, highFreqFactor, \
        originalContext, halfMode, positionOffset, index, loadFn, storeFn \
    ); \
}

#define ROPE_PAIRS_KERNEL(name, scalarType, loadFn, storeFn) \
extern "C" __global__ void name( \
    scalarType* out, \
    const scalarType* input, \
    const float* cosBuffer, \
    const float* sinBuffer, \
    unsigned int halfDim \
) { \
    unsigned int pairIndex = blockIdx.x * blockDim.x + threadIdx.x; \
    if (pairIndex >= halfDim) { \
        return; \
    } \
    unsigned int evenIndex = pairIndex * 2u; \
    unsigned int oddIndex = evenIndex + 1u; \
    float cosTheta = cosBuffer[pairIndex]; \
    float sinTheta = sinBuffer[pairIndex]; \
    rope_apply_pair(input, out, evenIndex, oddIndex, cosTheta, sinTheta, loadFn, storeFn); \
}

ROPE_KERNEL(rope_float32, float, rope_load_f32, rope_store_f32)
ROPE_KERNEL(rope_float16, __half, rope_load_f16, rope_store_f16)
ROPE_KERNEL(rope_bfloat16, __nv_bfloat16, rope_load_bf16, rope_store_bf16)

ROPE_PAIRS_KERNEL(rope_pairs_float32, float, rope_load_f32, rope_store_f32)
ROPE_PAIRS_KERNEL(rope_pairs_float16, __half, rope_load_f16, rope_store_f16)
ROPE_PAIRS_KERNEL(rope_pairs_bfloat16, __nv_bfloat16, rope_load_bf16, rope_store_bf16)

#endif
