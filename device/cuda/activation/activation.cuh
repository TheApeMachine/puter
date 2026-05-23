#ifndef PUTER_DEVICE_CUDA_ACTIVATION_ACTIVATION_CUH
#define PUTER_DEVICE_CUDA_ACTIVATION_ACTIVATION_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

static __device__ __forceinline__ float activation_selu_scale(void) {
    return 1.05070098735548049342f;
}

static __device__ __forceinline__ float activation_selu_alpha(void) {
    return 1.67326324235437728482f;
}

static __device__ __forceinline__ float activation_leaky_relu_slope(void) {
    return 0.01f;
}

static __device__ __forceinline__ float activation_gelu(float value) {
    return 0.5f * value * (1.0f + erff(value * 0.70710678118654752440f));
}

static __device__ __forceinline__ float activation_gelu_tanh(float value) {
    float cube = value * value * value;
    float inner = 0.7978845608028654f * (value + 0.044715f * cube);
    return 0.5f * value * (1.0f + tanhf(inner));
}

static __device__ __forceinline__ float activation_silu(float value) {
    return value / (1.0f + expf(-value));
}

static __device__ __forceinline__ float activation_sigmoid(float value) {
    return 1.0f / (1.0f + expf(-value));
}

static __device__ __forceinline__ float activation_relu(float value) {
    return value > 0.0f ? value : 0.0f;
}

static __device__ __forceinline__ __half activation_zero_h(void) {
    return __float2half(0.0f);
}

static __device__ __forceinline__ __half activation_one_h(void) {
    return __float2half(1.0f);
}

static __device__ __forceinline__ __half activation_silu_h1(__half value) {
    __half one = activation_one_h();
    return __hdiv(value, __hadd(one, hexp(__hneg(value))));
}

static __device__ __forceinline__ __half activation_sigmoid_h1(__half value) {
    __half one = activation_one_h();
    return __hdiv(one, __hadd(one, hexp(__hneg(value))));
}

static __device__ __forceinline__ __half activation_relu_h1(__half value) {
    return __hmax(value, activation_zero_h());
}

static __device__ __forceinline__ __half activation_gelu_tanh_h1(__half value) {
    __half alpha = __float2half(0.7978845608028654f);
    __half beta = __float2half(0.044715f);
    __half half = __float2half(0.5f);
    __half one = activation_one_h();
    __half cube = __hmul(__hmul(value, value), value);
    __half inner = __hmul(alpha, __hadd(value, __hmul(beta, cube)));
    return __hmul(half, __hmul(value, __hadd(one, htanh(inner))));
}

static __device__ __forceinline__ __half activation_gelu_h1(__half value) {
    return activation_gelu_tanh_h1(value);
}

static __device__ __forceinline__ __half2 activation_map_h2(__half2 value, __half (*operation)(__half)) {
    return __halves2half2(
        operation(__low2half(value)),
        operation(__high2half(value))
    );
}

static __device__ __forceinline__ __nv_bfloat16 activation_zero_bf16(void) {
    return __float2bfloat16(0.0f);
}

static __device__ __forceinline__ __nv_bfloat16 activation_one_bf16(void) {
    return __float2bfloat16(1.0f);
}

static __device__ __forceinline__ __nv_bfloat16 activation_bf16_exp(__nv_bfloat16 value) {
    return __hexp(value);
}

static __device__ __forceinline__ __nv_bfloat16 activation_bf16_log(__nv_bfloat16 value) {
    return __hlog(value);
}

static __device__ __forceinline__ __nv_bfloat16 activation_bf16_tanh(__nv_bfloat16 value) {
    return __htanh(value);
}

static __device__ __forceinline__ __nv_bfloat16 activation_silu_bf16(__nv_bfloat16 value) {
    __nv_bfloat16 one = activation_one_bf16();
    return __hdiv(value, __hadd(one, activation_bf16_exp(__hneg(value))));
}

static __device__ __forceinline__ __nv_bfloat16 activation_sigmoid_bf16(__nv_bfloat16 value) {
    __nv_bfloat16 one = activation_one_bf16();
    return __hdiv(one, __hadd(one, activation_bf16_exp(__hneg(value))));
}

static __device__ __forceinline__ __nv_bfloat16 activation_relu_bf16(__nv_bfloat16 value) {
    return __hmax(value, activation_zero_bf16());
}

static __device__ __forceinline__ __nv_bfloat16 activation_gelu_tanh_bf16(__nv_bfloat16 value) {
    __nv_bfloat16 alpha = __float2bfloat16(0.7978845608028654f);
    __nv_bfloat16 beta = __float2bfloat16(0.044715f);
    __nv_bfloat16 half = __float2bfloat16(0.5f);
    __nv_bfloat16 one = activation_one_bf16();
    __nv_bfloat16 cube = __hmul(__hmul(value, value), value);
    __nv_bfloat16 inner = __hmul(alpha, __hadd(value, __hmul(beta, cube)));
    return __hmul(half, __hmul(value, __hadd(one, activation_bf16_tanh(inner))));
}

static __device__ __forceinline__ __nv_bfloat16 activation_gelu_bf16(__nv_bfloat16 value) {
    return activation_gelu_tanh_bf16(value);
}

static __device__ __forceinline__ __nv_bfloat162 activation_map_bf162(
    __nv_bfloat162 value,
    __nv_bfloat16 (*operation)(__nv_bfloat16)
) {
    return __halves2bfloat162(
        operation(__low2bfloat16(value)),
        operation(__high2bfloat16(value))
    );
}

#endif
