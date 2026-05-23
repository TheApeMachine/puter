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

static __device__ __forceinline__ float4 param_snake_f4(float4 value, float alpha) {
    float sineX = sinf(alpha * value.x);
    float sineY = sinf(alpha * value.y);
    float sineZ = sinf(alpha * value.z);
    float sineW = sinf(alpha * value.w);
    float invAlpha = 1.0f / alpha;

    return make_float4(
        value.x + invAlpha * sineX * sineX,
        value.y + invAlpha * sineY * sineY,
        value.z + invAlpha * sineZ * sineZ,
        value.w + invAlpha * sineW * sineW
    );
}

static __device__ __forceinline__ float param_snake_f1(float value, float alpha) {
    float sine = sinf(alpha * value);
    return value + (1.0f / alpha) * sine * sine;
}

static __device__ __forceinline__ float4 param_hard_tanh_range_f4(float4 value, float minVal, float maxVal) {
    return make_float4(
        fminf(fmaxf(value.x, minVal), maxVal),
        fminf(fmaxf(value.y, minVal), maxVal),
        fminf(fmaxf(value.z, minVal), maxVal),
        fminf(fmaxf(value.w, minVal), maxVal)
    );
}

static __device__ __forceinline__ float param_hard_tanh_range_f1(float value, float minVal, float maxVal) {
    return fminf(fmaxf(value, minVal), maxVal);
}

static __device__ __forceinline__ float4 param_snake_parametric_f4(float4 value, float alpha, float beta) {
    float sineX = sinf(alpha * value.x);
    float sineY = sinf(alpha * value.y);
    float sineZ = sinf(alpha * value.z);
    float sineW = sinf(alpha * value.w);
    float invBeta = 1.0f / beta;

    return make_float4(
        value.x + invBeta * sineX * sineX,
        value.y + invBeta * sineY * sineY,
        value.z + invBeta * sineZ * sineZ,
        value.w + invBeta * sineW * sineW
    );
}

static __device__ __forceinline__ float param_snake_parametric_f1(float value, float alpha, float beta) {
    float sine = sinf(alpha * value);
    return value + (1.0f / beta) * sine * sine;
}

static __device__ __forceinline__ unsigned int param_rrelu_advance_state(unsigned int state) {
    return state * 1664525u + 1013904223u;
}

static __device__ __forceinline__ float param_rrelu_slope(unsigned int state, float lower, float upper) {
    return lower + (float(state >> 8) / 16777215.0f) * (upper - lower);
}

static __device__ __forceinline__ float param_rrelu_value(
    const float* input,
    unsigned int index,
    unsigned int count,
    float lower,
    float upper
) {
    if (input[index] > 0.0f) {
        return input[index];
    }

    unsigned int state = 0xA5A5A5A5u ^ __float_as_uint(lower) ^ __float_as_uint(upper);

    for (unsigned int prior = 0u; prior < index; prior++) {
        if (input[prior] <= 0.0f) {
            state = param_rrelu_advance_state(state);
        }
    }

    state = param_rrelu_advance_state(state);

    return input[index] * param_rrelu_slope(state, lower, upper);
}

#define DUAL_PARAM_KERNEL_F32(name, op_f4, op_f1) \
extern "C" __global__ void name##_float32( \
    const float* inputRaw, \
    float* outputRaw, \
    unsigned int count, \
    float param0, \
    float param1 \
) { \
    const float4* inputVector = reinterpret_cast<const float4*>(inputRaw); \
    float4* outputVector = reinterpret_cast<float4*>(outputRaw); \
    unsigned int vectorIndex = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int base = vectorIndex * 4u; \
    if (base + 3u < count) { \
        outputVector[vectorIndex] = op_f4(inputVector[vectorIndex], param0, param1); \
        return; \
    } \
    for (unsigned int offset = 0u; offset < 4u; offset++) { \
        unsigned int scalarIndex = base + offset; \
        if (scalarIndex < count) { \
            outputRaw[scalarIndex] = op_f1(inputRaw[scalarIndex], param0, param1); \
        } \
    } \
}

#define DUAL_PARAM_KERNEL_F16(name, op_f1) \
extern "C" __global__ void name##_float16( \
    const __half* input, \
    __half* output, \
    unsigned int count, \
    float param0, \
    float param1 \
) { \
    unsigned int pairIndex = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int base = pairIndex * 2u; \
    if (base + 1u < count) { \
        output[base] = __float2half(op_f1(__half2float(input[base]), param0, param1)); \
        output[base + 1u] = __float2half(op_f1(__half2float(input[base + 1u]), param0, param1)); \
        return; \
    } \
    if (base < count) { \
        output[base] = __float2half(op_f1(__half2float(input[base]), param0, param1)); \
    } \
}

#define DUAL_PARAM_KERNEL_BF16(name, op_f1) \
extern "C" __global__ void name##_bfloat16( \
    const __nv_bfloat16* input, \
    __nv_bfloat16* output, \
    unsigned int count, \
    float param0, \
    float param1 \
) { \
    unsigned int pairIndex = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int base = pairIndex * 2u; \
    if (base + 1u < count) { \
        output[base] = __float2bfloat16(op_f1(__bfloat162float(input[base]), param0, param1)); \
        output[base + 1u] = __float2bfloat16(op_f1(__bfloat162float(input[base + 1u]), param0, param1)); \
        return; \
    } \
    if (base < count) { \
        output[base] = __float2bfloat16(op_f1(__bfloat162float(input[base]), param0, param1)); \
    } \
}

#define RRELU_KERNEL_F32(name) \
extern "C" __global__ void name##_float32( \
    const float* input, \
    float* output, \
    unsigned int count, \
    float lower, \
    float upper \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (index >= count) { \
        return; \
    } \
    output[index] = param_rrelu_value(input, index, count, lower, upper); \
}

PARAMETRIC_KERNEL_F32(snake, param_snake_f4, param_snake_f1)

static __device__ __forceinline__ __half param_snake_h1(__half value, __half alpha) {
    float alphaF = __half2float(alpha);
    float valueF = __half2float(value);
    float sine = sinf(alphaF * valueF);

    return __float2half(valueF + (1.0f / alphaF) * sine * sine);
}

static __device__ __forceinline__ __half2 param_snake_h2(__half2 value, __half alpha) {
    return __halves2half2(
        param_snake_h1(__low2half(value), alpha),
        param_snake_h1(__high2half(value), alpha)
    );
}

static __device__ __forceinline__ __nv_bfloat16 param_snake_bf16(__nv_bfloat16 value, __nv_bfloat16 alpha) {
    float alphaF = __bfloat162float(alpha);
    float valueF = __bfloat162float(value);
    float sine = sinf(alphaF * valueF);

    return __float2bfloat16(valueF + (1.0f / alphaF) * sine * sine);
}

static __device__ __forceinline__ __nv_bfloat162 param_snake_bf162(__nv_bfloat162 value, __nv_bfloat16 alpha) {
    return __halves2bfloat162(
        param_snake_bf16(__low2bfloat16(value), alpha),
        param_snake_bf16(__high2bfloat16(value), alpha)
    );
}

PARAMETRIC_KERNEL_F16(snake, param_snake_h2, param_snake_h1)
PARAMETRIC_KERNEL_BF16(snake, param_snake_bf162, param_snake_bf16)

DUAL_PARAM_KERNEL_F32(hard_tanh_range, param_hard_tanh_range_f4, param_hard_tanh_range_f1)
DUAL_PARAM_KERNEL_F16(hard_tanh_range, param_hard_tanh_range_f1)
DUAL_PARAM_KERNEL_BF16(hard_tanh_range, param_hard_tanh_range_f1)

DUAL_PARAM_KERNEL_F32(snake_parametric, param_snake_parametric_f4, param_snake_parametric_f1)
DUAL_PARAM_KERNEL_F16(snake_parametric, param_snake_parametric_f1)
DUAL_PARAM_KERNEL_BF16(snake_parametric, param_snake_parametric_f1)

RRELU_KERNEL_F32(rrelu)

static __device__ __forceinline__ float param_rrelu_value_half(
    const __half* input,
    unsigned int index,
    unsigned int count,
    float lower,
    float upper
) {
    float value = __half2float(input[index]);

    if (value > 0.0f) {
        return value;
    }

    unsigned int state = 0xA5A5A5A5u ^ __float_as_uint(lower) ^ __float_as_uint(upper);

    for (unsigned int prior = 0u; prior < index; prior++) {
        if (__half2float(input[prior]) <= 0.0f) {
            state = param_rrelu_advance_state(state);
        }
    }

    state = param_rrelu_advance_state(state);

    return value * param_rrelu_slope(state, lower, upper);
}

static __device__ __forceinline__ float param_rrelu_value_bfloat16(
    const __nv_bfloat16* input,
    unsigned int index,
    unsigned int count,
    float lower,
    float upper
) {
    float value = __bfloat162float(input[index]);

    if (value > 0.0f) {
        return value;
    }

    unsigned int state = 0xA5A5A5A5u ^ __float_as_uint(lower) ^ __float_as_uint(upper);

    for (unsigned int prior = 0u; prior < index; prior++) {
        if (__bfloat162float(input[prior]) <= 0.0f) {
            state = param_rrelu_advance_state(state);
        }
    }

    state = param_rrelu_advance_state(state);

    return value * param_rrelu_slope(state, lower, upper);
}

extern "C" __global__ void rrelu_float16(
    const __half* input,
    __half* output,
    unsigned int count,
    float lower,
    float upper
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= count) {
        return;
    }

    output[index] = __float2half(param_rrelu_value_half(input, index, count, lower, upper));
}

extern "C" __global__ void rrelu_bfloat16(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    unsigned int count,
    float lower,
    float upper
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= count) {
        return;
    }

    output[index] = __float2bfloat16(param_rrelu_value_bfloat16(input, index, count, lower, upper));
}
