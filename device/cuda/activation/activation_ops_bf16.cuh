#ifndef PUTER_DEVICE_CUDA_ACTIVATION_ACTIVATION_OPS_BF16_CUH
#define PUTER_DEVICE_CUDA_ACTIVATION_ACTIVATION_OPS_BF16_CUH

#include "activation.cuh"

static __device__ __forceinline__ __nv_bfloat162 activation_relu_b2(__nv_bfloat162 value) {
    return activation_map_bf162(value, activation_relu_bf16);
}

static __device__ __forceinline__ __nv_bfloat162 activation_exp_b2(__nv_bfloat162 value) {
    return __h2exp(value);
}

static __device__ __forceinline__ __nv_bfloat16 activation_exp_b1(__nv_bfloat16 value) {
    return activation_bf16_exp(value);
}

static __device__ __forceinline__ __nv_bfloat162 activation_log_b2(__nv_bfloat162 value) {
    return __h2log(value);
}

static __device__ __forceinline__ __nv_bfloat16 activation_log_b1(__nv_bfloat16 value) {
    return activation_bf16_log(value);
}

static __device__ __forceinline__ __nv_bfloat162 activation_tanh_b2(__nv_bfloat162 value) {
    return __h2tanh(value);
}

static __device__ __forceinline__ __nv_bfloat16 activation_tanh_b1(__nv_bfloat16 value) {
    return activation_bf16_tanh(value);
}

static __device__ __forceinline__ __nv_bfloat162 activation_gelu_b2(__nv_bfloat162 value) {
    return activation_map_bf162(value, activation_gelu_bf16);
}

static __device__ __forceinline__ __nv_bfloat162 activation_sigmoid_b2(__nv_bfloat162 value) {
    return activation_map_bf162(value, activation_sigmoid_bf16);
}

static __device__ __forceinline__ __nv_bfloat162 activation_silu_b2(__nv_bfloat162 value) {
    return activation_map_bf162(value, activation_silu_bf16);
}

static __device__ __forceinline__ __nv_bfloat16 activation_softsign_b1(__nv_bfloat16 value) {
    __nv_bfloat16 one = activation_one_bf16();
    return __hdiv(value, __hadd(one, __habs(value)));
}

static __device__ __forceinline__ __nv_bfloat162 activation_softsign_b2(__nv_bfloat162 value) {
    return activation_map_bf162(value, activation_softsign_b1);
}

static __device__ __forceinline__ __nv_bfloat16 activation_elu_b1(__nv_bfloat16 value) {
    __nv_bfloat16 zero = activation_zero_bf16();
    __nv_bfloat16 one = activation_one_bf16();

    if (__hgt(value, zero)) {
        return value;
    }

    return __hsub(activation_bf16_exp(value), one);
}

static __device__ __forceinline__ __nv_bfloat162 activation_elu_b2(__nv_bfloat162 value) {
    return activation_map_bf162(value, activation_elu_b1);
}

static __device__ __forceinline__ __nv_bfloat16 activation_selu_b1(__nv_bfloat16 value) {
    __nv_bfloat16 scale = __float2bfloat16(activation_selu_scale());
    __nv_bfloat16 alpha = __float2bfloat16(activation_selu_alpha());
    __nv_bfloat16 zero = activation_zero_bf16();
    __nv_bfloat16 one = activation_one_bf16();

    if (__hgt(value, zero)) {
        return __hmul(scale, value);
    }

    return __hmul(scale, __hmul(alpha, __hsub(activation_bf16_exp(value), one)));
}

static __device__ __forceinline__ __nv_bfloat162 activation_selu_b2(__nv_bfloat162 value) {
    return activation_map_bf162(value, activation_selu_b1);
}

static __device__ __forceinline__ __nv_bfloat16 activation_leaky_relu_b1(__nv_bfloat16 value) {
    __nv_bfloat16 slope = __float2bfloat16(activation_leaky_relu_slope());
    __nv_bfloat16 zero = activation_zero_bf16();
    return __hgt(value, zero) ? value : __hmul(slope, value);
}

static __device__ __forceinline__ __nv_bfloat162 activation_leaky_relu_b2(__nv_bfloat162 value) {
    return activation_map_bf162(value, activation_leaky_relu_b1);
}

static __device__ __forceinline__ __nv_bfloat16 activation_hardsigmoid_b1(__nv_bfloat16 value) {
    __nv_bfloat16 zero = activation_zero_bf16();
    __nv_bfloat16 one = activation_one_bf16();
    __nv_bfloat16 six = __float2bfloat16(6.0f);
    __nv_bfloat16 half = __float2bfloat16(0.5f);
    __nv_bfloat16 scaled = __hadd(__hdiv(value, six), half);
    return __hmin(one, __hmax(zero, scaled));
}

static __device__ __forceinline__ __nv_bfloat162 activation_hardsigmoid_b2(__nv_bfloat162 value) {
    return activation_map_bf162(value, activation_hardsigmoid_b1);
}

static __device__ __forceinline__ __nv_bfloat16 activation_hardswish_b1(__nv_bfloat16 value) {
    __nv_bfloat16 three = __float2bfloat16(3.0f);
    __nv_bfloat16 six = __float2bfloat16(6.0f);
    __nv_bfloat16 zero = activation_zero_bf16();
    __nv_bfloat16 one = activation_one_bf16();
    __nv_bfloat16 inner = __hdiv(__hadd(value, three), six);
    return __hmul(value, __hmin(one, __hmax(zero, inner)));
}

static __device__ __forceinline__ __nv_bfloat162 activation_hardswish_b2(__nv_bfloat162 value) {
    return activation_map_bf162(value, activation_hardswish_b1);
}

static __device__ __forceinline__ __nv_bfloat16 activation_log1p_b1(__nv_bfloat16 value) {
    return activation_bf16_log(__hadd(activation_one_bf16(), value));
}

static __device__ __forceinline__ __nv_bfloat162 activation_log1p_b2(__nv_bfloat162 value) {
    return activation_map_bf162(value, activation_log1p_b1);
}

static __device__ __forceinline__ __nv_bfloat16 activation_expm1_b1(__nv_bfloat16 value) {
    return __hsub(activation_bf16_exp(value), activation_one_bf16());
}

static __device__ __forceinline__ __nv_bfloat162 activation_expm1_b2(__nv_bfloat162 value) {
    return activation_map_bf162(value, activation_expm1_b1);
}

static __device__ __forceinline__ __nv_bfloat16 activation_celu_b1(__nv_bfloat16 value) {
    __nv_bfloat16 alpha = activation_one_bf16();
    __nv_bfloat16 zero = activation_zero_bf16();

    if (__hgt(value, zero)) {
        return value;
    }

    return __hmul(alpha, __hsub(activation_bf16_exp(__hdiv(value, alpha)), activation_one_bf16()));
}

static __device__ __forceinline__ __nv_bfloat162 activation_celu_b2(__nv_bfloat162 value) {
    return activation_map_bf162(value, activation_celu_b1);
}

static __device__ __forceinline__ __nv_bfloat16 activation_softplus_b1(__nv_bfloat16 value) {
    __nv_bfloat16 limit = __float2bfloat16(20.0f);

    if (__hgt(value, limit)) {
        return value;
    }

    return activation_bf16_log(__hadd(activation_one_bf16(), activation_bf16_exp(value)));
}

static __device__ __forceinline__ __nv_bfloat162 activation_softplus_b2(__nv_bfloat162 value) {
    return activation_map_bf162(value, activation_softplus_b1);
}

static __device__ __forceinline__ __nv_bfloat16 activation_mish_b1(__nv_bfloat16 value) {
    return __hmul(value, activation_bf16_tanh(activation_softplus_b1(value)));
}

static __device__ __forceinline__ __nv_bfloat162 activation_mish_b2(__nv_bfloat162 value) {
    return activation_map_bf162(value, activation_mish_b1);
}

static __device__ __forceinline__ __nv_bfloat16 activation_log_sigmoid_b1(__nv_bfloat16 value) {
    __nv_bfloat16 negValue = __hneg(value);
    __nv_bfloat16 limit = __float2bfloat16(20.0f);
    __nv_bfloat16 softplus = __hgt(negValue, limit)
        ? negValue
        : activation_bf16_log(__hadd(activation_one_bf16(), activation_bf16_exp(negValue)));
    return __hneg(softplus);
}

static __device__ __forceinline__ __nv_bfloat162 activation_log_sigmoid_b2(__nv_bfloat162 value) {
    return activation_map_bf162(value, activation_log_sigmoid_b1);
}

static __device__ __forceinline__ __nv_bfloat162 activation_gelu_tanh_b2(__nv_bfloat162 value) {
    return activation_map_bf162(value, activation_gelu_tanh_bf16);
}

static __device__ __forceinline__ __nv_bfloat16 activation_hardtanh_b1(__nv_bfloat16 value) {
    __nv_bfloat16 minValue = __float2bfloat16(-1.0f);
    __nv_bfloat16 maxValue = __float2bfloat16(1.0f);
    return __hmin(maxValue, __hmax(minValue, value));
}

static __device__ __forceinline__ __nv_bfloat162 activation_hardtanh_b2(__nv_bfloat162 value) {
    return activation_map_bf162(value, activation_hardtanh_b1);
}

static __device__ __forceinline__ __nv_bfloat16 activation_hard_gelu_b1(__nv_bfloat16 value) {
    __nv_bfloat16 three = __float2bfloat16(3.0f);
    __nv_bfloat16 six = __float2bfloat16(6.0f);
    __nv_bfloat16 zero = activation_zero_bf16();
    __nv_bfloat16 inner = __hadd(value, three);
    inner = __hmax(zero, __hmin(inner, six));
    return __hmul(value, __hdiv(inner, six));
}

static __device__ __forceinline__ __nv_bfloat162 activation_hard_gelu_b2(__nv_bfloat162 value) {
    return activation_map_bf162(value, activation_hard_gelu_b1);
}

static __device__ __forceinline__ __nv_bfloat16 activation_quick_gelu_b1(__nv_bfloat16 value) {
    __nv_bfloat16 scale = __float2bfloat16(1.702f);
    __nv_bfloat16 one = activation_one_bf16();
    return __hdiv(value, __hadd(one, activation_bf16_exp(__hneg(__hmul(scale, value)))));
}

static __device__ __forceinline__ __nv_bfloat162 activation_quick_gelu_b2(__nv_bfloat162 value) {
    return activation_map_bf162(value, activation_quick_gelu_b1);
}

static __device__ __forceinline__ __nv_bfloat16 activation_tanh_shrink_b1(__nv_bfloat16 value) {
    return __hsub(value, activation_bf16_tanh(value));
}

static __device__ __forceinline__ __nv_bfloat162 activation_tanh_shrink_b2(__nv_bfloat162 value) {
    return activation_map_bf162(value, activation_tanh_shrink_b1);
}

#endif
