#include "activation.cuh"

#define ACTIVATION_UNARY_KERNEL(name, expr) \
extern "C" __global__ void name##_float32( \
    const float* input, \
    float* output, \
    unsigned int count \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (index >= count) { \
        return; \
    } \
    float value = input[index]; \
    output[index] = (expr); \
}

ACTIVATION_UNARY_KERNEL(exp, expf(value))
ACTIVATION_UNARY_KERNEL(log, logf(value))
ACTIVATION_UNARY_KERNEL(log1p, log1pf(value))
ACTIVATION_UNARY_KERNEL(expm1, expm1f(value))
ACTIVATION_UNARY_KERNEL(sigmoid, 1.0f / (1.0f + expf(-value)))
ACTIVATION_UNARY_KERNEL(log_sigmoid, -logf(1.0f + expf(-value)))
ACTIVATION_UNARY_KERNEL(tanh, tanhf(value))
ACTIVATION_UNARY_KERNEL(silu, value / (1.0f + expf(-value)))
ACTIVATION_UNARY_KERNEL(swish, value / (1.0f + expf(-value)))
ACTIVATION_UNARY_KERNEL(softsign, value / (1.0f + fabsf(value)))
ACTIVATION_UNARY_KERNEL(softplus, log1pf(expf(value)))
ACTIVATION_UNARY_KERNEL(mish, value * tanhf(log1pf(expf(value))))
ACTIVATION_UNARY_KERNEL(hardsigmoid, fminf(1.0f, fmaxf(0.0f, 0.2f * value + 0.5f)))
ACTIVATION_UNARY_KERNEL(hardswish, value * fminf(1.0f, fmaxf(0.0f, value + 3.0f)) / 6.0f))
ACTIVATION_UNARY_KERNEL(hardtanh, fminf(1.0f, fmaxf(-1.0f, value)))
ACTIVATION_UNARY_KERNEL(quick_gelu, value * sigmoidf(1.702f * value))
ACTIVATION_UNARY_KERNEL(tanh_shrink, value - tanhf(value))

extern "C" __global__ void gelu_float32(
    const float* input,
    float* output,
    unsigned int count
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= count) {
        return;
    }

    float value = input[index];
    output[index] = 0.5f * value * (1.0f + erff(value * 0.70710678118654752440f));
}

extern "C" __global__ void gelu_tanh_float32(
    const float* input,
    float* output,
    unsigned int count
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= count) {
        return;
    }

    float value = input[index];
    float cube = value * value * value;
    float inner = 0.7978845608028654f * (value + 0.044715f * cube);
    output[index] = 0.5f * value * (1.0f + tanhf(inner));
}

extern "C" __global__ void hard_gelu_float32(
    const float* input,
    float* output,
    unsigned int count
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= count) {
        return;
    }

    float value = input[index];
    float scaled = (value + 1.5f) / 3.0f;

    if (value <= -1.5f) {
        output[index] = 0.0f;
        return;
    }

    if (value >= 1.5f) {
        output[index] = value;
        return;
    }

    output[index] = value * scaled;
}

extern "C" __global__ void relu_float32(
    const float* input,
    float* output,
    unsigned int count
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= count) {
        return;
    }

    float value = input[index];
    output[index] = value > 0.0f ? value : 0.0f;
}

extern "C" __global__ void leaky_relu_float32(
    const float* input,
    float* output,
    unsigned int count
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= count) {
        return;
    }

    float value = input[index];
    float slope = activation_leaky_relu_slope();
    output[index] = value > 0.0f ? value : slope * value;
}

extern "C" __global__ void elu_float32(
    const float* input,
    float* output,
    unsigned int count
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= count) {
        return;
    }

    float value = input[index];
    output[index] = value > 0.0f ? value : expf(value) - 1.0f;
}

extern "C" __global__ void celu_float32(
    const float* input,
    float* output,
    unsigned int count
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= count) {
        return;
    }

    float value = input[index];
    output[index] = value > 0.0f ? value : expf(value) - 1.0f;
}

extern "C" __global__ void selu_float32(
    const float* input,
    float* output,
    unsigned int count
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= count) {
        return;
    }

    float value = input[index];
    float scale = activation_selu_scale();
    float alpha = activation_selu_alpha();
    output[index] = value > 0.0f ? scale * value : scale * alpha * (expf(value) - 1.0f);
}

#define ACTIVATION_UNARY_KERNEL_F16(name, expr) \
extern "C" __global__ void name##_float16( \
    const __half* input, \
    __half* output, \
    unsigned int count \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (index >= count) { \
        return; \
    } \
    float value = __half2float(input[index]); \
    output[index] = __float2half(expr); \
}

ACTIVATION_UNARY_KERNEL_F16(exp, expf(value))
ACTIVATION_UNARY_KERNEL_F16(relu, value > 0.0f ? value : 0.0f)

#define ACTIVATION_UNARY_KERNEL_BF16(name, expr) \
extern "C" __global__ void name##_bfloat16( \
    const __nv_bfloat16* input, \
    __nv_bfloat16* output, \
    unsigned int count \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (index >= count) { \
        return; \
    } \
    float value = activation_bf16_to_float(input[index]); \
    output[index] = activation_float_to_bf16(expr); \
}

ACTIVATION_UNARY_KERNEL_BF16(exp, expf(value))
ACTIVATION_UNARY_KERNEL_BF16(relu, value > 0.0f ? value : 0.0f)
