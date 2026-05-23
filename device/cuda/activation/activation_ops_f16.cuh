#ifndef PUTER_DEVICE_CUDA_ACTIVATION_ACTIVATION_OPS_F16_CUH
#define PUTER_DEVICE_CUDA_ACTIVATION_ACTIVATION_OPS_F16_CUH

#include "activation.cuh"

static __device__ __forceinline__ __half2 activation_relu_h2(__half2 value) {
    return activation_map_h2(value, activation_relu_h1);
}

static __device__ __forceinline__ __half activation_exp_h1(__half value) {
    return hexp(value);
}

static __device__ __forceinline__ __half2 activation_exp_h2(__half2 value) {
    return hexp2(value);
}

static __device__ __forceinline__ __half activation_log_h1(__half value) {
    return hlog(value);
}

static __device__ __forceinline__ __half2 activation_log_h2(__half2 value) {
    return hlog2(value);
}

static __device__ __forceinline__ __half activation_tanh_h1(__half value) {
    return htanh(value);
}

static __device__ __forceinline__ __half2 activation_tanh_h2(__half2 value) {
    return htanh2(value);
}

static __device__ __forceinline__ __half2 activation_gelu_h2(__half2 value) {
    return activation_map_h2(value, activation_gelu_h1);
}

static __device__ __forceinline__ __half2 activation_sigmoid_h2(__half2 value) {
    return activation_map_h2(value, activation_sigmoid_h1);
}

static __device__ __forceinline__ __half2 activation_silu_h2(__half2 value) {
    return activation_map_h2(value, activation_silu_h1);
}

static __device__ __forceinline__ __half activation_softsign_h1(__half value) {
    __half one = activation_one_h();
    return __hdiv(value, __hadd(one, __habs(value)));
}

static __device__ __forceinline__ __half2 activation_softsign_h2(__half2 value) {
    return activation_map_h2(value, activation_softsign_h1);
}

static __device__ __forceinline__ __half activation_elu_h1(__half value) {
    __half zero = activation_zero_h();
    __half one = activation_one_h();

    if (__hgt(value, zero)) {
        return value;
    }

    return __hsub(hexp(value), one);
}

static __device__ __forceinline__ __half2 activation_elu_h2(__half2 value) {
    return activation_map_h2(value, activation_elu_h1);
}

static __device__ __forceinline__ __half activation_selu_h1(__half value) {
    __half scale = __float2half(activation_selu_scale());
    __half alpha = __float2half(activation_selu_alpha());
    __half zero = activation_zero_h();
    __half one = activation_one_h();

    if (__hgt(value, zero)) {
        return __hmul(scale, value);
    }

    return __hmul(scale, __hmul(alpha, __hsub(hexp(value), one)));
}

static __device__ __forceinline__ __half2 activation_selu_h2(__half2 value) {
    return activation_map_h2(value, activation_selu_h1);
}

static __device__ __forceinline__ __half activation_leaky_relu_h1(__half value) {
    __half slope = __float2half(activation_leaky_relu_slope());
    __half zero = activation_zero_h();
    return __hgt(value, zero) ? value : __hmul(slope, value);
}

static __device__ __forceinline__ __half2 activation_leaky_relu_h2(__half2 value) {
    return activation_map_h2(value, activation_leaky_relu_h1);
}

static __device__ __forceinline__ __half activation_hardsigmoid_h1(__half value) {
    __half zero = activation_zero_h();
    __half one = activation_one_h();
    __half six = __float2half(6.0f);
    __half half = __float2half(0.5f);
    __half scaled = __hadd(__hdiv(value, six), half);
    return __hmin(one, __hmax(zero, scaled));
}

static __device__ __forceinline__ __half2 activation_hardsigmoid_h2(__half2 value) {
    return activation_map_h2(value, activation_hardsigmoid_h1);
}

static __device__ __forceinline__ __half activation_hardswish_h1(__half value) {
    __half three = __float2half(3.0f);
    __half six = __float2half(6.0f);
    __half zero = activation_zero_h();
    __half one = activation_one_h();
    __half inner = __hdiv(__hadd(value, three), six);
    return __hmul(value, __hmin(one, __hmax(zero, inner)));
}

static __device__ __forceinline__ __half2 activation_hardswish_h2(__half2 value) {
    return activation_map_h2(value, activation_hardswish_h1);
}

static __device__ __forceinline__ __half activation_log1p_h1(__half value) {
    return hlog(__hadd(activation_one_h(), value));
}

static __device__ __forceinline__ __half2 activation_log1p_h2(__half2 value) {
    return activation_map_h2(value, activation_log1p_h1);
}

static __device__ __forceinline__ __half activation_expm1_h1(__half value) {
    return __hsub(hexp(value), activation_one_h());
}

static __device__ __forceinline__ __half2 activation_expm1_h2(__half2 value) {
    return activation_map_h2(value, activation_expm1_h1);
}

static __device__ __forceinline__ __half activation_celu_h1(__half value) {
    __half alpha = activation_one_h();
    __half zero = activation_zero_h();

    if (__hgt(value, zero)) {
        return value;
    }

    return __hmul(alpha, __hsub(hexp(__hdiv(value, alpha)), activation_one_h()));
}

static __device__ __forceinline__ __half2 activation_celu_h2(__half2 value) {
    return activation_map_h2(value, activation_celu_h1);
}

static __device__ __forceinline__ __half activation_softplus_h1(__half value) {
    __half limit = __float2half(20.0f);

    if (__hgt(value, limit)) {
        return value;
    }

    return hlog(__hadd(activation_one_h(), hexp(value)));
}

static __device__ __forceinline__ __half2 activation_softplus_h2(__half2 value) {
    return activation_map_h2(value, activation_softplus_h1);
}

static __device__ __forceinline__ __half activation_mish_h1(__half value) {
    return __hmul(value, htanh(activation_softplus_h1(value)));
}

static __device__ __forceinline__ __half2 activation_mish_h2(__half2 value) {
    return activation_map_h2(value, activation_mish_h1);
}

static __device__ __forceinline__ __half activation_log_sigmoid_h1(__half value) {
    __half negValue = __hneg(value);
    __half limit = __float2half(20.0f);
    __half softplus = __hgt(negValue, limit) ? negValue : hlog(__hadd(activation_one_h(), hexp(negValue)));
    return __hneg(softplus);
}

static __device__ __forceinline__ __half2 activation_log_sigmoid_h2(__half2 value) {
    return activation_map_h2(value, activation_log_sigmoid_h1);
}

static __device__ __forceinline__ __half2 activation_gelu_tanh_h2(__half2 value) {
    return activation_map_h2(value, activation_gelu_tanh_h1);
}

static __device__ __forceinline__ __half activation_hardtanh_h1(__half value) {
    __half minValue = __float2half(-1.0f);
    __half maxValue = __float2half(1.0f);
    return __hmin(maxValue, __hmax(minValue, value));
}

static __device__ __forceinline__ __half2 activation_hardtanh_h2(__half2 value) {
    return activation_map_h2(value, activation_hardtanh_h1);
}

static __device__ __forceinline__ __half activation_hard_gelu_h1(__half value) {
    __half three = __float2half(3.0f);
    __half six = __float2half(6.0f);
    __half zero = activation_zero_h();
    __half inner = __hadd(value, three);
    inner = __hmax(zero, __hmin(inner, six));
    return __hmul(value, __hdiv(inner, six));
}

static __device__ __forceinline__ __half2 activation_hard_gelu_h2(__half2 value) {
    return activation_map_h2(value, activation_hard_gelu_h1);
}

static __device__ __forceinline__ __half activation_quick_gelu_h1(__half value) {
    __half scale = __float2half(1.702f);
    __half one = activation_one_h();
    return __hdiv(value, __hadd(one, hexp(__hneg(__hmul(scale, value)))));
}

static __device__ __forceinline__ __half2 activation_quick_gelu_h2(__half2 value) {
    return activation_map_h2(value, activation_quick_gelu_h1);
}

static __device__ __forceinline__ __half activation_tanh_shrink_h1(__half value) {
    return __hsub(value, htanh(value));
}

static __device__ __forceinline__ __half2 activation_tanh_shrink_h2(__half2 value) {
    return activation_map_h2(value, activation_tanh_shrink_h1);
}

#endif
