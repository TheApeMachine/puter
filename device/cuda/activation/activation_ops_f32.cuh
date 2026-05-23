#ifndef PUTER_DEVICE_CUDA_ACTIVATION_ACTIVATION_OPS_F32_CUH
#define PUTER_DEVICE_CUDA_ACTIVATION_ACTIVATION_OPS_F32_CUH

#include "activation.cuh"

static __device__ __forceinline__ float4 activation_relu_f4(float4 value) {
    return make_float4(
        fmaxf(0.0f, value.x),
        fmaxf(0.0f, value.y),
        fmaxf(0.0f, value.z),
        fmaxf(0.0f, value.w)
    );
}

static __device__ __forceinline__ float activation_relu_f1(float value) {
    return fmaxf(0.0f, value);
}

static __device__ __forceinline__ float4 activation_exp_f4(float4 value) {
    return make_float4(expf(value.x), expf(value.y), expf(value.z), expf(value.w));
}

static __device__ __forceinline__ float activation_exp_f1(float value) {
    return expf(value);
}

static __device__ __forceinline__ float4 activation_log_f4(float4 value) {
    return make_float4(logf(value.x), logf(value.y), logf(value.z), logf(value.w));
}

static __device__ __forceinline__ float activation_log_f1(float value) {
    return logf(value);
}

static __device__ __forceinline__ float4 activation_tanh_f4(float4 value) {
    return make_float4(tanhf(value.x), tanhf(value.y), tanhf(value.z), tanhf(value.w));
}

static __device__ __forceinline__ float activation_tanh_f1(float value) {
    return tanhf(value);
}

static __device__ __forceinline__ float4 activation_gelu_f4(float4 value) {
    return make_float4(
        activation_gelu(value.x),
        activation_gelu(value.y),
        activation_gelu(value.z),
        activation_gelu(value.w)
    );
}

static __device__ __forceinline__ float activation_gelu_f1(float value) {
    return activation_gelu(value);
}

static __device__ __forceinline__ float4 activation_sigmoid_f4(float4 value) {
    return make_float4(
        activation_sigmoid(value.x),
        activation_sigmoid(value.y),
        activation_sigmoid(value.z),
        activation_sigmoid(value.w)
    );
}

static __device__ __forceinline__ float activation_sigmoid_f1(float value) {
    return activation_sigmoid(value);
}

static __device__ __forceinline__ float4 activation_silu_f4(float4 value) {
    return make_float4(
        activation_silu(value.x),
        activation_silu(value.y),
        activation_silu(value.z),
        activation_silu(value.w)
    );
}

static __device__ __forceinline__ float activation_silu_f1(float value) {
    return activation_silu(value);
}

static __device__ __forceinline__ float4 activation_softsign_f4(float4 value) {
    return make_float4(
        value.x / (1.0f + fabsf(value.x)),
        value.y / (1.0f + fabsf(value.y)),
        value.z / (1.0f + fabsf(value.z)),
        value.w / (1.0f + fabsf(value.w))
    );
}

static __device__ __forceinline__ float activation_softsign_f1(float value) {
    return value / (1.0f + fabsf(value));
}

static __device__ __forceinline__ float4 activation_elu_f4(float4 value) {
    return make_float4(
        value.x > 0.0f ? value.x : expf(value.x) - 1.0f,
        value.y > 0.0f ? value.y : expf(value.y) - 1.0f,
        value.z > 0.0f ? value.z : expf(value.z) - 1.0f,
        value.w > 0.0f ? value.w : expf(value.w) - 1.0f
    );
}

static __device__ __forceinline__ float activation_elu_f1(float value) {
    return value > 0.0f ? value : expf(value) - 1.0f;
}

static __device__ __forceinline__ float4 activation_selu_f4(float4 value) {
    float scale = activation_selu_scale();
    float alpha = activation_selu_alpha();

    return make_float4(
        value.x > 0.0f ? scale * value.x : scale * alpha * (expf(value.x) - 1.0f),
        value.y > 0.0f ? scale * value.y : scale * alpha * (expf(value.y) - 1.0f),
        value.z > 0.0f ? scale * value.z : scale * alpha * (expf(value.z) - 1.0f),
        value.w > 0.0f ? scale * value.w : scale * alpha * (expf(value.w) - 1.0f)
    );
}

static __device__ __forceinline__ float activation_selu_f1(float value) {
    float scale = activation_selu_scale();
    float alpha = activation_selu_alpha();
    return value > 0.0f ? scale * value : scale * alpha * (expf(value) - 1.0f);
}

static __device__ __forceinline__ float4 activation_leaky_relu_f4(float4 value) {
    float slope = activation_leaky_relu_slope();

    return make_float4(
        value.x > 0.0f ? value.x : slope * value.x,
        value.y > 0.0f ? value.y : slope * value.y,
        value.z > 0.0f ? value.z : slope * value.z,
        value.w > 0.0f ? value.w : slope * value.w
    );
}

static __device__ __forceinline__ float activation_leaky_relu_f1(float value) {
    float slope = activation_leaky_relu_slope();
    return value > 0.0f ? value : slope * value;
}

static __device__ __forceinline__ float4 activation_hardsigmoid_f4(float4 value) {
    return make_float4(
        fminf(1.0f, fmaxf(0.0f, value.x / 6.0f + 0.5f)),
        fminf(1.0f, fmaxf(0.0f, value.y / 6.0f + 0.5f)),
        fminf(1.0f, fmaxf(0.0f, value.z / 6.0f + 0.5f)),
        fminf(1.0f, fmaxf(0.0f, value.w / 6.0f + 0.5f))
    );
}

static __device__ __forceinline__ float activation_hardsigmoid_f1(float value) {
    return fminf(1.0f, fmaxf(0.0f, value / 6.0f + 0.5f));
}

static __device__ __forceinline__ float4 activation_hardswish_f4(float4 value) {
    return make_float4(
        value.x * fminf(1.0f, fmaxf(0.0f, (value.x + 3.0f) / 6.0f)),
        value.y * fminf(1.0f, fmaxf(0.0f, (value.y + 3.0f) / 6.0f)),
        value.z * fminf(1.0f, fmaxf(0.0f, (value.z + 3.0f) / 6.0f)),
        value.w * fminf(1.0f, fmaxf(0.0f, (value.w + 3.0f) / 6.0f))
    );
}

static __device__ __forceinline__ float activation_hardswish_f1(float value) {
    return value * fminf(1.0f, fmaxf(0.0f, (value + 3.0f) / 6.0f));
}

static __device__ __forceinline__ float4 activation_log1p_f4(float4 value) {
    return make_float4(log1pf(value.x), log1pf(value.y), log1pf(value.z), log1pf(value.w));
}

static __device__ __forceinline__ float activation_log1p_f1(float value) {
    return log1pf(value);
}

static __device__ __forceinline__ float4 activation_expm1_f4(float4 value) {
    return make_float4(expm1f(value.x), expm1f(value.y), expm1f(value.z), expm1f(value.w));
}

static __device__ __forceinline__ float activation_expm1_f1(float value) {
    return expm1f(value);
}

static __device__ __forceinline__ float4 activation_celu_f4(float4 value) {
    const float alpha = 1.0f;

    return make_float4(
        value.x > 0.0f ? value.x : alpha * (expf(value.x / alpha) - 1.0f),
        value.y > 0.0f ? value.y : alpha * (expf(value.y / alpha) - 1.0f),
        value.z > 0.0f ? value.z : alpha * (expf(value.z / alpha) - 1.0f),
        value.w > 0.0f ? value.w : alpha * (expf(value.w / alpha) - 1.0f)
    );
}

static __device__ __forceinline__ float activation_celu_f1(float value) {
    const float alpha = 1.0f;
    return value > 0.0f ? value : alpha * (expf(value / alpha) - 1.0f);
}

static __device__ __forceinline__ float activation_softplus_f1(float value) {
    return value > 20.0f ? value : log1pf(expf(value));
}

static __device__ __forceinline__ float4 activation_softplus_f4(float4 value) {
    return make_float4(
        activation_softplus_f1(value.x),
        activation_softplus_f1(value.y),
        activation_softplus_f1(value.z),
        activation_softplus_f1(value.w)
    );
}

static __device__ __forceinline__ float activation_mish_f1(float value) {
    float softplus = activation_softplus_f1(value);
    return value * tanhf(softplus);
}

static __device__ __forceinline__ float4 activation_mish_f4(float4 value) {
    return make_float4(
        activation_mish_f1(value.x),
        activation_mish_f1(value.y),
        activation_mish_f1(value.z),
        activation_mish_f1(value.w)
    );
}

static __device__ __forceinline__ float activation_log_sigmoid_f1(float value) {
    float softplus = -value > 20.0f ? -value : log1pf(expf(-value));
    return -softplus;
}

static __device__ __forceinline__ float4 activation_log_sigmoid_f4(float4 value) {
    return make_float4(
        activation_log_sigmoid_f1(value.x),
        activation_log_sigmoid_f1(value.y),
        activation_log_sigmoid_f1(value.z),
        activation_log_sigmoid_f1(value.w)
    );
}

static __device__ __forceinline__ float4 activation_gelu_tanh_f4(float4 value) {
    return make_float4(
        activation_gelu_tanh(value.x),
        activation_gelu_tanh(value.y),
        activation_gelu_tanh(value.z),
        activation_gelu_tanh(value.w)
    );
}

static __device__ __forceinline__ float activation_gelu_tanh_f1(float value) {
    return activation_gelu_tanh(value);
}

static __device__ __forceinline__ float4 activation_hardtanh_f4(float4 value) {
    return make_float4(
        fminf(1.0f, fmaxf(-1.0f, value.x)),
        fminf(1.0f, fmaxf(-1.0f, value.y)),
        fminf(1.0f, fmaxf(-1.0f, value.z)),
        fminf(1.0f, fmaxf(-1.0f, value.w))
    );
}

static __device__ __forceinline__ float activation_hardtanh_f1(float value) {
    return fminf(1.0f, fmaxf(-1.0f, value));
}

static __device__ __forceinline__ float activation_hard_gelu_f1(float value) {
    float inner = value + 3.0f;

    if (inner < 0.0f) {
        inner = 0.0f;
    }

    if (inner > 6.0f) {
        inner = 6.0f;
    }

    return value * (inner / 6.0f);
}

static __device__ __forceinline__ float4 activation_hard_gelu_f4(float4 value) {
    return make_float4(
        activation_hard_gelu_f1(value.x),
        activation_hard_gelu_f1(value.y),
        activation_hard_gelu_f1(value.z),
        activation_hard_gelu_f1(value.w)
    );
}

static __device__ __forceinline__ float activation_quick_gelu_f1(float value) {
    return value / (1.0f + expf(-1.702f * value));
}

static __device__ __forceinline__ float4 activation_quick_gelu_f4(float4 value) {
    return make_float4(
        activation_quick_gelu_f1(value.x),
        activation_quick_gelu_f1(value.y),
        activation_quick_gelu_f1(value.z),
        activation_quick_gelu_f1(value.w)
    );
}

static __device__ __forceinline__ float4 activation_tanh_shrink_f4(float4 value) {
    return make_float4(
        value.x - tanhf(value.x),
        value.y - tanhf(value.y),
        value.z - tanhf(value.z),
        value.w - tanhf(value.w)
    );
}

static __device__ __forceinline__ float activation_tanh_shrink_f1(float value) {
    return value - tanhf(value);
}

#endif
