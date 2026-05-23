#ifndef PUTER_DEVICE_CUDA_ATTENTION_MASKING_CUH
#define PUTER_DEVICE_CUDA_ATTENTION_MASKING_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

static __device__ __forceinline__ float masking_bf16_to_float(unsigned short value) {
    __nv_bfloat16 raw;
    memcpy(&raw, &value, sizeof(raw));
    return __bfloat162float(raw);
}

static __device__ __forceinline__ unsigned short masking_float_to_bf16(float value) {
    return __bfloat16_as_ushort(__float2bfloat16(value));
}

static __device__ __forceinline__ float4 masking_bf16_to_float4(ushort4 value) {
    return make_float4(
        masking_bf16_to_float(value.x),
        masking_bf16_to_float(value.y),
        masking_bf16_to_float(value.z),
        masking_bf16_to_float(value.w)
    );
}

static __device__ __forceinline__ ushort4 masking_float4_to_bf16(float4 value) {
    return make_ushort4(
        masking_float_to_bf16(value.x),
        masking_float_to_bf16(value.y),
        masking_float_to_bf16(value.z),
        masking_float_to_bf16(value.w)
    );
}

static __device__ __forceinline__ float masking_neg_inf_float32() {
    return __int_as_float(0xFF800000u);
}

static __device__ __forceinline__ __half masking_neg_inf_float16() {
    return __ushort_as_half(static_cast<unsigned short>(0xFC00u));
}

static __device__ __forceinline__ unsigned short masking_neg_inf_bfloat16() {
    return masking_float_to_bf16(masking_neg_inf_float32());
}

#define APPLY_MASK_KERNEL(name, scalarType, loadVecFn, storeVecFn, loadScalarFn, storeScalarFn) \
extern "C" __global__ void name( \
    const scalarType* input, \
    const scalarType* mask, \
    scalarType* out, \
    unsigned int count \
) { \
    unsigned int vectorIndex = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int base = vectorIndex * 4u; \
    if (base + 3u < count) { \
        storeVecFn(out, base, loadVecFn(input, base) + loadVecFn(mask, base)); \
        return; \
    } \
    for (unsigned int offset = 0; offset < 4u; offset++) { \
        unsigned int scalarIndex = base + offset; \
        if (scalarIndex < count) { \
            storeScalarFn(out, scalarIndex, loadScalarFn(input, scalarIndex) + loadScalarFn(mask, scalarIndex)); \
        } \
    } \
}

static __device__ __forceinline__ float4 masking_load_f32_vec(const float* values, unsigned int base) {
    return *reinterpret_cast<const float4*>(values + base);
}

static __device__ __forceinline__ void masking_store_f32_vec(float* values, unsigned int base, float4 value) {
    *reinterpret_cast<float4*>(values + base) = value;
}

static __device__ __forceinline__ float masking_load_f32_scalar(const float* values, unsigned int index) {
    return values[index];
}

static __device__ __forceinline__ void masking_store_f32_scalar(float* values, unsigned int index, float value) {
    values[index] = value;
}

static __device__ __forceinline__ float4 masking_load_f16_vec(const __half* values, unsigned int base) {
    float2 low = __half22float2(*reinterpret_cast<const half2*>(values + base));
    float2 high = __half22float2(*reinterpret_cast<const half2*>(values + base + 2));
    return make_float4(low.x, low.y, high.x, high.y);
}

static __device__ __forceinline__ void masking_store_f16_vec(__half* values, unsigned int base, float4 value) {
    half2 low = __float22half2_rn(make_float2(value.x, value.y));
    half2 high = __float22half2_rn(make_float2(value.z, value.w));
    *reinterpret_cast<half2*>(values + base) = low;
    *reinterpret_cast<half2*>(values + base + 2) = high;
}

static __device__ __forceinline__ float masking_load_f16_scalar(const __half* values, unsigned int index) {
    return __half2float(values[index]);
}

static __device__ __forceinline__ void masking_store_f16_scalar(__half* values, unsigned int index, float value) {
    values[index] = __float2half(value);
}

static __device__ __forceinline__ float4 masking_load_bf16_vec(const __nv_bfloat16* values, unsigned int base) {
    return masking_bf16_to_float4(*reinterpret_cast<const ushort4*>(reinterpret_cast<const unsigned short*>(values) + base));
}

static __device__ __forceinline__ void masking_store_bf16_vec(__nv_bfloat16* values, unsigned int base, float4 value) {
    *reinterpret_cast<ushort4*>(reinterpret_cast<unsigned short*>(values) + base) = masking_float4_to_bf16(value);
}

static __device__ __forceinline__ float masking_load_bf16_scalar(const __nv_bfloat16* values, unsigned int index) {
    return __bfloat162float(values[index]);
}

static __device__ __forceinline__ void masking_store_bf16_scalar(__nv_bfloat16* values, unsigned int index, float value) {
    values[index] = __float2bfloat16(value);
}

APPLY_MASK_KERNEL(
    apply_mask_float32, float,
    masking_load_f32_vec, masking_store_f32_vec,
    masking_load_f32_scalar, masking_store_f32_scalar
)
APPLY_MASK_KERNEL(
    apply_mask_float16, __half,
    masking_load_f16_vec, masking_store_f16_vec,
    masking_load_f16_scalar, masking_store_f16_scalar
)
APPLY_MASK_KERNEL(
    apply_mask_bfloat16, __nv_bfloat16,
    masking_load_bf16_vec, masking_store_bf16_vec,
    masking_load_bf16_scalar, masking_store_bf16_scalar
)

#define CAUSAL_MASK_KERNEL(name, scalarType, negInfFn, storeVecFn, storeScalarFn) \
extern "C" __global__ void name( \
    scalarType* out, \
    unsigned int rows, \
    unsigned int cols \
) { \
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x; \
    if (row >= rows) { \
        return; \
    } \
    unsigned int rowBase = row * cols; \
    scalarType negInf = negInfFn(); \
    scalarType zero = scalarType(0); \
    float rowValue = static_cast<float>(row); \
    for (unsigned int colBase = 0; colBase < cols; colBase += 4u) { \
        if (colBase + 3u < cols) { \
            float4 colIdx = make_float4( \
                static_cast<float>(colBase), \
                static_cast<float>(colBase + 1u), \
                static_cast<float>(colBase + 2u), \
                static_cast<float>(colBase + 3u) \
            ); \
            float4 result = make_float4( \
                colIdx.x > rowValue ? static_cast<float>(negInf) : 0.0f, \
                colIdx.y > rowValue ? static_cast<float>(negInf) : 0.0f, \
                colIdx.z > rowValue ? static_cast<float>(negInf) : 0.0f, \
                colIdx.w > rowValue ? static_cast<float>(negInf) : 0.0f \
            ); \
            storeVecFn(out, rowBase + colBase, result); \
            continue; \
        } \
        for (unsigned int offset = 0; offset < 4u; offset++) { \
            unsigned int col = colBase + offset; \
            if (col >= cols) { \
                return; \
            } \
            storeScalarFn(out, rowBase + col, col > row ? negInf : zero); \
        } \
    } \
}

static __device__ __forceinline__ void masking_store_f32_causal_vec(float* values, unsigned int base, float4 value) {
    *reinterpret_cast<float4*>(values + base) = value;
}

static __device__ __forceinline__ void masking_store_f32_causal_scalar(float* values, unsigned int index, float value) {
    values[index] = value;
}

static __device__ __forceinline__ void masking_store_f16_causal_vec(__half* values, unsigned int base, float4 value) {
    half2 low = __float22half2_rn(make_float2(value.x, value.y));
    half2 high = __float22half2_rn(make_float2(value.z, value.w));
    *reinterpret_cast<half2*>(values + base) = low;
    *reinterpret_cast<half2*>(values + base + 2) = high;
}

static __device__ __forceinline__ void masking_store_f16_causal_scalar(__half* values, unsigned int index, __half value) {
    values[index] = value;
}

static __device__ __forceinline__ void masking_store_bf16_causal_vec(__nv_bfloat16* values, unsigned int base, float4 value) {
    *reinterpret_cast<ushort4*>(reinterpret_cast<unsigned short*>(values) + base) = masking_float4_to_bf16(value);
}

static __device__ __forceinline__ void masking_store_bf16_causal_scalar(__nv_bfloat16* values, unsigned int index, __nv_bfloat16 value) {
    values[index] = value;
}

CAUSAL_MASK_KERNEL(causal_mask_float32, float, masking_neg_inf_float32, masking_store_f32_causal_vec, masking_store_f32_causal_scalar)
CAUSAL_MASK_KERNEL(causal_mask_float16, __half, masking_neg_inf_float16, masking_store_f16_causal_vec, masking_store_f16_causal_scalar)
CAUSAL_MASK_KERNEL(causal_mask_bfloat16, __nv_bfloat16, masking_neg_inf_bfloat16, masking_store_bf16_causal_vec, masking_store_bf16_causal_scalar)

#define ALIBI_BIAS_KERNEL(name, loadScoreVecFn, loadScoreScalarFn, storeScoreVecFn, storeScoreScalarFn, loadSlopeFn) \
extern "C" __global__ void name( \
    const float* scores, \
    const float* slope, \
    float* out, \
    unsigned int rows, \
    unsigned int cols \
) { \
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x; \
    if (row >= rows) { \
        return; \
    } \
    unsigned int rowBase = row * cols; \
    float slopeValue = loadSlopeFn(slope, 0u); \
    float rowValue = static_cast<float>(row); \
    for (unsigned int colBase = 0; colBase < cols; colBase += 4u) { \
        if (colBase + 3u < cols) { \
            float4 colIdx = make_float4( \
                static_cast<float>(colBase), \
                static_cast<float>(colBase + 1u), \
                static_cast<float>(colBase + 2u), \
                static_cast<float>(colBase + 3u) \
            ); \
            float4 score4 = loadScoreVecFn(scores, rowBase + colBase); \
            float4 result = make_float4( \
                colIdx.x <= rowValue ? score4.x - slopeValue * (rowValue - colIdx.x) : score4.x, \
                colIdx.y <= rowValue ? score4.y - slopeValue * (rowValue - colIdx.y) : score4.y, \
                colIdx.z <= rowValue ? score4.z - slopeValue * (rowValue - colIdx.z) : score4.z, \
                colIdx.w <= rowValue ? score4.w - slopeValue * (rowValue - colIdx.w) : score4.w \
            ); \
            storeScoreVecFn(out, rowBase + colBase, result); \
            continue; \
        } \
        for (unsigned int offset = 0; offset < 4u; offset++) { \
            unsigned int col = colBase + offset; \
            if (col >= cols) { \
                return; \
            } \
            unsigned int index = rowBase + col; \
            float scoreValue = loadScoreScalarFn(scores, index); \
            float outputValue = row >= col ? scoreValue - slopeValue * static_cast<float>(row - col) : scoreValue; \
            storeScoreScalarFn(out, index, outputValue); \
        } \
    } \
}

static __device__ __forceinline__ float4 masking_load_score_f32_vec(const float* values, unsigned int base) {
    return *reinterpret_cast<const float4*>(values + base);
}

static __device__ __forceinline__ float masking_load_score_f32_scalar(const float* values, unsigned int index) {
    return values[index];
}

static __device__ __forceinline__ void masking_store_score_f32_vec(float* values, unsigned int base, float4 value) {
    *reinterpret_cast<float4*>(values + base) = value;
}

static __device__ __forceinline__ void masking_store_score_f32_scalar(float* values, unsigned int index, float value) {
    values[index] = value;
}

static __device__ __forceinline__ float masking_load_slope_f32(const float* values, unsigned int index) {
    return values[index];
}

ALIBI_BIAS_KERNEL(
    alibi_bias_float32,
    masking_load_score_f32_vec,
    masking_load_score_f32_scalar,
    masking_store_score_f32_vec,
    masking_store_score_f32_scalar,
    masking_load_slope_f32
)

extern "C" __global__ void alibi_bias_float16(
    const __half* scores,
    const __half* slope,
    __half* out,
    unsigned int rows,
    unsigned int cols
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows) {
        return;
    }

    unsigned int rowBase = row * cols;
    float slopeValue = __half2float(slope[0]);
    float rowValue = static_cast<float>(row);

    for (unsigned int colBase = 0; colBase < cols; colBase += 4u) {
        if (colBase + 3u < cols) {
            float4 score4 = masking_load_f16_vec(scores, rowBase + colBase);
            float4 colIdx = make_float4(
                static_cast<float>(colBase),
                static_cast<float>(colBase + 1u),
                static_cast<float>(colBase + 2u),
                static_cast<float>(colBase + 3u)
            );
            float4 result = make_float4(
                colIdx.x <= rowValue ? score4.x - slopeValue * (rowValue - colIdx.x) : score4.x,
                colIdx.y <= rowValue ? score4.y - slopeValue * (rowValue - colIdx.y) : score4.y,
                colIdx.z <= rowValue ? score4.z - slopeValue * (rowValue - colIdx.z) : score4.z,
                colIdx.w <= rowValue ? score4.w - slopeValue * (rowValue - colIdx.w) : score4.w
            );
            masking_store_f16_vec(out, rowBase + colBase, result);
            continue;
        }

        for (unsigned int offset = 0; offset < 4u; offset++) {
            unsigned int col = colBase + offset;

            if (col >= cols) {
                return;
            }

            unsigned int index = rowBase + col;
            float scoreValue = __half2float(scores[index]);
            float outputValue = row >= col ? scoreValue - slopeValue * static_cast<float>(row - col) : scoreValue;
            out[index] = __float2half(outputValue);
        }
    }
}

extern "C" __global__ void alibi_bias_bfloat16(
    const __nv_bfloat16* scores,
    const __nv_bfloat16* slope,
    __nv_bfloat16* out,
    unsigned int rows,
    unsigned int cols
) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows) {
        return;
    }

    unsigned int rowBase = row * cols;
    float slopeValue = __bfloat162float(slope[0]);
    float rowValue = static_cast<float>(row);

    for (unsigned int colBase = 0; colBase < cols; colBase += 4u) {
        if (colBase + 3u < cols) {
            float4 score4 = masking_load_bf16_vec(scores, rowBase + colBase);
            float4 colIdx = make_float4(
                static_cast<float>(colBase),
                static_cast<float>(colBase + 1u),
                static_cast<float>(colBase + 2u),
                static_cast<float>(colBase + 3u)
            );
            float4 result = make_float4(
                colIdx.x <= rowValue ? score4.x - slopeValue * (rowValue - colIdx.x) : score4.x,
                colIdx.y <= rowValue ? score4.y - slopeValue * (rowValue - colIdx.y) : score4.y,
                colIdx.z <= rowValue ? score4.z - slopeValue * (rowValue - colIdx.z) : score4.z,
                colIdx.w <= rowValue ? score4.w - slopeValue * (rowValue - colIdx.w) : score4.w
            );
            masking_store_bf16_vec(out, rowBase + colBase, result);
            continue;
        }

        for (unsigned int offset = 0; offset < 4u; offset++) {
            unsigned int col = colBase + offset;

            if (col >= cols) {
                return;
            }

            unsigned int index = rowBase + col;
            float scoreValue = __bfloat162float(scores[index]);
            float outputValue = row >= col ? scoreValue - slopeValue * static_cast<float>(row - col) : scoreValue;
            out[index] = __float2bfloat16(outputValue);
        }
    }
}

#endif
