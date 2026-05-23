#include "activation.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

static __device__ __forceinline__ float4 param_prelu_f4(float4 value, float slope) {
    return make_float4(
        value.x > 0.0f ? value.x : slope * value.x,
        value.y > 0.0f ? value.y : slope * value.y,
        value.z > 0.0f ? value.z : slope * value.z,
        value.w > 0.0f ? value.w : slope * value.w
    );
}

static __device__ __forceinline__ float param_prelu_f1(float value, float slope) {
    return value > 0.0f ? value : slope * value;
}

static __device__ __forceinline__ float4 param_leaky_relu_f4(float4 value, float slope) {
    return param_prelu_f4(value, slope);
}

static __device__ __forceinline__ float param_leaky_relu_f1(float value, float slope) {
    return param_prelu_f1(value, slope);
}

static __device__ __forceinline__ float4 param_elu_f4(float4 value, float alpha) {
    return make_float4(
        value.x > 0.0f ? value.x : alpha * (expf(value.x) - 1.0f),
        value.y > 0.0f ? value.y : alpha * (expf(value.y) - 1.0f),
        value.z > 0.0f ? value.z : alpha * (expf(value.z) - 1.0f),
        value.w > 0.0f ? value.w : alpha * (expf(value.w) - 1.0f)
    );
}

static __device__ __forceinline__ float param_elu_f1(float value, float alpha) {
    return value > 0.0f ? value : alpha * (expf(value) - 1.0f);
}

static __device__ __forceinline__ float4 param_celu_f4(float4 value, float alpha) {
    return make_float4(
        value.x > 0.0f ? value.x : alpha * (expf(value.x / alpha) - 1.0f),
        value.y > 0.0f ? value.y : alpha * (expf(value.y / alpha) - 1.0f),
        value.z > 0.0f ? value.z : alpha * (expf(value.z / alpha) - 1.0f),
        value.w > 0.0f ? value.w : alpha * (expf(value.w / alpha) - 1.0f)
    );
}

static __device__ __forceinline__ float param_celu_f1(float value, float alpha) {
    return value > 0.0f ? value : alpha * (expf(value / alpha) - 1.0f);
}

static __device__ __forceinline__ float4 param_threshold_f4(float4 value, float threshold) {
    return make_float4(
        value.x > threshold ? value.x : 0.0f,
        value.y > threshold ? value.y : 0.0f,
        value.z > threshold ? value.z : 0.0f,
        value.w > threshold ? value.w : 0.0f
    );
}

static __device__ __forceinline__ float param_threshold_f1(float value, float threshold) {
    return value > threshold ? value : 0.0f;
}

static __device__ __forceinline__ __half param_prelu_h1(__half value, __half slope) {
    __half zero = __float2half(0.0f);
    return __hgt(value, zero) ? value : __hmul(slope, value);
}

static __device__ __forceinline__ __half2 param_prelu_h2(__half2 value, __half slope) {
    return __halves2half2(
        param_prelu_h1(__low2half(value), slope),
        param_prelu_h1(__high2half(value), slope)
    );
}

static __device__ __forceinline__ __half param_elu_h1(__half value, __half alpha) {
    __half zero = __float2half(0.0f);
    __half one = __float2half(1.0f);

    if (__hgt(value, zero)) {
        return value;
    }

    return __hmul(alpha, __hsub(hexp(value), one));
}

static __device__ __forceinline__ __half2 param_elu_h2(__half2 value, __half alpha) {
    return __halves2half2(
        param_elu_h1(__low2half(value), alpha),
        param_elu_h1(__high2half(value), alpha)
    );
}

static __device__ __forceinline__ __half param_celu_h1(__half value, __half alpha) {
    __half zero = __float2half(0.0f);
    __half one = __float2half(1.0f);

    if (__hgt(value, zero)) {
        return value;
    }

    return __hmul(alpha, __hsub(hexp(__hdiv(value, alpha)), one));
}

static __device__ __forceinline__ __half2 param_celu_h2(__half2 value, __half alpha) {
    return __halves2half2(
        param_celu_h1(__low2half(value), alpha),
        param_celu_h1(__high2half(value), alpha)
    );
}

static __device__ __forceinline__ __half param_threshold_h1(__half value, __half threshold) {
    __half zero = __float2half(0.0f);
    return __hgt(value, threshold) ? value : zero;
}

static __device__ __forceinline__ __half2 param_threshold_h2(__half2 value, __half threshold) {
    return __halves2half2(
        param_threshold_h1(__low2half(value), threshold),
        param_threshold_h1(__high2half(value), threshold)
    );
}

static __device__ __forceinline__ __nv_bfloat16 param_prelu_bf16(__nv_bfloat16 value, __nv_bfloat16 slope) {
    __nv_bfloat16 zero = __float2bfloat16(0.0f);
    return __hgt(value, zero) ? value : __hmul(slope, value);
}

static __device__ __forceinline__ __nv_bfloat162 param_prelu_bf162(__nv_bfloat162 value, __nv_bfloat16 slope) {
    return __halves2bfloat162(
        param_prelu_bf16(__low2bfloat16(value), slope),
        param_prelu_bf16(__high2bfloat16(value), slope)
    );
}

static __device__ __forceinline__ __nv_bfloat16 param_elu_bf16(__nv_bfloat16 value, __nv_bfloat16 alpha) {
    __nv_bfloat16 zero = __float2bfloat16(0.0f);
    __nv_bfloat16 one = __float2bfloat16(1.0f);

    if (__hgt(value, zero)) {
        return value;
    }

    return __hmul(alpha, __hsub(activation_bf16_exp(value), one));
}

static __device__ __forceinline__ __nv_bfloat162 param_elu_bf162(__nv_bfloat162 value, __nv_bfloat16 alpha) {
    return __halves2bfloat162(
        param_elu_bf16(__low2bfloat16(value), alpha),
        param_elu_bf16(__high2bfloat16(value), alpha)
    );
}

static __device__ __forceinline__ __nv_bfloat16 param_celu_bf16(__nv_bfloat16 value, __nv_bfloat16 alpha) {
    __nv_bfloat16 zero = __float2bfloat16(0.0f);
    __nv_bfloat16 one = __float2bfloat16(1.0f);

    if (__hgt(value, zero)) {
        return value;
    }

    return __hmul(alpha, __hsub(activation_bf16_exp(__hdiv(value, alpha)), one));
}

static __device__ __forceinline__ __nv_bfloat162 param_celu_bf162(__nv_bfloat162 value, __nv_bfloat16 alpha) {
    return __halves2bfloat162(
        param_celu_bf16(__low2bfloat16(value), alpha),
        param_celu_bf16(__high2bfloat16(value), alpha)
    );
}

static __device__ __forceinline__ __nv_bfloat16 param_threshold_bf16(__nv_bfloat16 value, __nv_bfloat16 threshold) {
    __nv_bfloat16 zero = __float2bfloat16(0.0f);
    return __hgt(value, threshold) ? value : zero;
}

static __device__ __forceinline__ __nv_bfloat162 param_threshold_bf162(__nv_bfloat162 value, __nv_bfloat16 threshold) {
    return __halves2bfloat162(
        param_threshold_bf16(__low2bfloat16(value), threshold),
        param_threshold_bf16(__high2bfloat16(value), threshold)
    );
}

#define PARAMETRIC_KERNEL_F32(name, op_f4, op_f1) \
extern "C" __global__ void name##_float32( \
    const float* inputRaw, \
    float* outputRaw, \
    unsigned int count, \
    float param \
) { \
    const float4* inputVector = reinterpret_cast<const float4*>(inputRaw); \
    float4* outputVector = reinterpret_cast<float4*>(outputRaw); \
    unsigned int vectorIndex = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int base = vectorIndex * 4u; \
    if (base + 3u < count) { \
        outputVector[vectorIndex] = op_f4(inputVector[vectorIndex], param); \
        return; \
    } \
    for (unsigned int offset = 0u; offset < 4u; offset++) { \
        unsigned int scalarIndex = base + offset; \
        if (scalarIndex < count) { \
            outputRaw[scalarIndex] = op_f1(inputRaw[scalarIndex], param); \
        } \
    } \
}

#define PARAMETRIC_KERNEL_F16(name, op_h2, op_h1) \
extern "C" __global__ void name##_float16( \
    const __half* input, \
    __half* output, \
    unsigned int count, \
    float param \
) { \
    __half paramHalf = __float2half(param); \
    unsigned int pairIndex = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int base = pairIndex * 2u; \
    if (base + 1u < count) { \
        __half2 value = *reinterpret_cast<const __half2*>(&input[base]); \
        *reinterpret_cast<__half2*>(&output[base]) = op_h2(value, paramHalf); \
        return; \
    } \
    if (base < count) { \
        output[base] = op_h1(input[base], paramHalf); \
    } \
}

#define PARAMETRIC_KERNEL_BF16(name, op_b2, op_b1) \
extern "C" __global__ void name##_bfloat16( \
    const __nv_bfloat16* input, \
    __nv_bfloat16* output, \
    unsigned int count, \
    float param \
) { \
    __nv_bfloat16 paramBf16 = __float2bfloat16(param); \
    unsigned int pairIndex = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int base = pairIndex * 2u; \
    if (base + 1u < count) { \
        __nv_bfloat162 value = *reinterpret_cast<const __nv_bfloat162*>(&input[base]); \
        *reinterpret_cast<__nv_bfloat162*>(&output[base]) = op_b2(value, paramBf16); \
        return; \
    } \
    if (base < count) { \
        output[base] = op_b1(input[base], paramBf16); \
    } \
}

PARAMETRIC_KERNEL_F32(prelu_slope, param_prelu_f4, param_prelu_f1)
PARAMETRIC_KERNEL_F32(leaky_relu_slope, param_leaky_relu_f4, param_leaky_relu_f1)
PARAMETRIC_KERNEL_F32(elu_alpha, param_elu_f4, param_elu_f1)
PARAMETRIC_KERNEL_F32(celu_alpha, param_celu_f4, param_celu_f1)
PARAMETRIC_KERNEL_F32(threshold, param_threshold_f4, param_threshold_f1)

PARAMETRIC_KERNEL_F16(prelu_slope, param_prelu_h2, param_prelu_h1)
PARAMETRIC_KERNEL_F16(leaky_relu_slope, param_prelu_h2, param_prelu_h1)
PARAMETRIC_KERNEL_F16(elu_alpha, param_elu_h2, param_elu_h1)
PARAMETRIC_KERNEL_F16(celu_alpha, param_celu_h2, param_celu_h1)
PARAMETRIC_KERNEL_F16(threshold, param_threshold_h2, param_threshold_h1)

PARAMETRIC_KERNEL_BF16(prelu_slope, param_prelu_bf162, param_prelu_bf16)
PARAMETRIC_KERNEL_BF16(leaky_relu_slope, param_prelu_bf162, param_prelu_bf16)
PARAMETRIC_KERNEL_BF16(elu_alpha, param_elu_bf162, param_elu_bf16)
PARAMETRIC_KERNEL_BF16(celu_alpha, param_celu_bf162, param_celu_bf16)
PARAMETRIC_KERNEL_BF16(threshold, param_threshold_bf162, param_threshold_bf16)

static __device__ __forceinline__ float param_prelu_v_f1(float value, float slope) {
    return value > 0.0f ? value : slope * value;
}

static __device__ __forceinline__ float4 param_prelu_v_f4(float4 value, float4 slopes) {
    return make_float4(
        param_prelu_v_f1(value.x, slopes.x),
        param_prelu_v_f1(value.y, slopes.y),
        param_prelu_v_f1(value.z, slopes.z),
        param_prelu_v_f1(value.w, slopes.w)
    );
}

static __device__ __forceinline__ __half param_prelu_v_h1(__half value, __half slope) {
    __half zero = __float2half(0.0f);
    return __hgt(value, zero) ? value : __hmul(slope, value);
}

static __device__ __forceinline__ __half2 param_prelu_v_h2(__half2 value, __half2 slopes) {
    return __halves2half2(
        param_prelu_v_h1(__low2half(value), __low2half(slopes)),
        param_prelu_v_h1(__high2half(value), __high2half(slopes))
    );
}

static __device__ __forceinline__ __nv_bfloat16 param_prelu_v_bf16(__nv_bfloat16 value, __nv_bfloat16 slope) {
    __nv_bfloat16 zero = __float2bfloat16(0.0f);
    return __hgt(value, zero) ? value : __hmul(slope, value);
}

static __device__ __forceinline__ __nv_bfloat162 param_prelu_v_bf162(__nv_bfloat162 value, __nv_bfloat162 slopes) {
    return __halves2bfloat162(
        param_prelu_v_bf16(__low2bfloat16(value), __low2bfloat16(slopes)),
        param_prelu_v_bf16(__high2bfloat16(value), __high2bfloat16(slopes))
    );
}

#define INDEXED_PARAMETRIC_KERNEL_F32(name) \
extern "C" __global__ void name##_float32( \
    const float* input, \
    const float* slopes, \
    float* output, \
    unsigned int count \
) { \
    const float4* inputVector = reinterpret_cast<const float4*>(input); \
    const float4* slopeVector = reinterpret_cast<const float4*>(slopes); \
    float4* outputVector = reinterpret_cast<float4*>(output); \
    unsigned int vectorIndex = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int base = vectorIndex * 4u; \
    if (base + 3u < count) { \
        outputVector[vectorIndex] = param_prelu_v_f4(inputVector[vectorIndex], slopeVector[vectorIndex]); \
        return; \
    } \
    for (unsigned int offset = 0u; offset < 4u; offset++) { \
        unsigned int scalarIndex = base + offset; \
        if (scalarIndex < count) { \
            output[scalarIndex] = param_prelu_v_f1(input[scalarIndex], slopes[scalarIndex]); \
        } \
    } \
}

#define INDEXED_PARAMETRIC_KERNEL_F16(name) \
extern "C" __global__ void name##_float16( \
    const __half* input, \
    const __half* slopes, \
    __half* output, \
    unsigned int count \
) { \
    unsigned int pairIndex = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int base = pairIndex * 2u; \
    if (base + 1u < count) { \
        __half2 value = *reinterpret_cast<const __half2*>(&input[base]); \
        __half2 slope = *reinterpret_cast<const __half2*>(&slopes[base]); \
        *reinterpret_cast<__half2*>(&output[base]) = param_prelu_v_h2(value, slope); \
        return; \
    } \
    if (base < count) { \
        output[base] = param_prelu_v_h1(input[base], slopes[base]); \
    } \
}

#define INDEXED_PARAMETRIC_KERNEL_BF16(name) \
extern "C" __global__ void name##_bfloat16( \
    const __nv_bfloat16* input, \
    const __nv_bfloat16* slopes, \
    __nv_bfloat16* output, \
    unsigned int count \
) { \
    unsigned int pairIndex = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int base = pairIndex * 2u; \
    if (base + 1u < count) { \
        __nv_bfloat162 value = *reinterpret_cast<const __nv_bfloat162*>(&input[base]); \
        __nv_bfloat162 slope = *reinterpret_cast<const __nv_bfloat162*>(&slopes[base]); \
        *reinterpret_cast<__nv_bfloat162*>(&output[base]) = param_prelu_v_bf162(value, slope); \
        return; \
    } \
    if (base < count) { \
        output[base] = param_prelu_v_bf16(input[base], slopes[base]); \
    } \
}

INDEXED_PARAMETRIC_KERNEL_F32(prelu_v)
INDEXED_PARAMETRIC_KERNEL_F16(prelu_v)
INDEXED_PARAMETRIC_KERNEL_BF16(prelu_v)
