#include <metal_stdlib>
#include "elementwise_gelu_f64.metalinc"

using namespace metal;

constant float metalSELUAlpha = 1.67326324235437728482f;
constant float metalSELUScale = 1.05070098735548049342f;
constant float metalLeakyReLUSlope = 0.01f;
static inline float extended_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort extended_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

static inline float4 extended_bf16_to_float4(ushort4 value) {
    return float4(
        extended_bf16_to_float(value.x),
        extended_bf16_to_float(value.y),
        extended_bf16_to_float(value.z),
        extended_bf16_to_float(value.w)
    );
}

static inline ushort4 extended_float4_to_bf16(float4 value) {
    return ushort4(
        extended_float_to_bf16(value.x),
        extended_float_to_bf16(value.y),
        extended_float_to_bf16(value.z),
        extended_float_to_bf16(value.w)
    );
}

template <typename UnaryOp>
static inline void extended_unary_float32(
    device const float4* inputVector,
    device float4* outVector,
    constant uint& count,
    uint index [[thread_position_in_grid]],
    UnaryOp op
) {
    uint base = index * 4;

    if (base + 3 < count) {
        outVector[index] = op(inputVector[index]);
        return;
    }

    device const float* input = reinterpret_cast<device const float*>(inputVector);
    device float* out = reinterpret_cast<device float*>(outVector);

    for (uint offset = 0; offset < 4; offset++) {
        uint scalarIndex = base + offset;

        if (scalarIndex < count) {
            out[scalarIndex] = op(input[scalarIndex]);
        }
    }
}

template <typename UnaryOp>
static inline void extended_unary_float16(
    device const half4* inputVector,
    device half4* outVector,
    constant uint& count,
    uint index [[thread_position_in_grid]],
    UnaryOp op
) {
    uint base = index * 4;

    if (base + 3 < count) {
        outVector[index] = half4(op(float4(inputVector[index])));
        return;
    }

    device const half* input = reinterpret_cast<device const half*>(inputVector);
    device half* out = reinterpret_cast<device half*>(outVector);

    for (uint offset = 0; offset < 4; offset++) {
        uint scalarIndex = base + offset;

        if (scalarIndex < count) {
            out[scalarIndex] = half(op(float(input[scalarIndex])));
        }
    }
}

template <typename UnaryOp>
static inline void extended_unary_bfloat16(
    device const ushort4* inputVector,
    device ushort4* outVector,
    constant uint& count,
    uint index [[thread_position_in_grid]],
    UnaryOp op
) {
    uint base = index * 4;

    if (base + 3 < count) {
        outVector[index] = extended_float4_to_bf16(op(extended_bf16_to_float4(inputVector[index])));
        return;
    }

    device const ushort* input = reinterpret_cast<device const ushort*>(inputVector);
    device ushort* out = reinterpret_cast<device ushort*>(outVector);

    for (uint offset = 0; offset < 4; offset++) {
        uint scalarIndex = base + offset;

        if (scalarIndex < count) {
            out[scalarIndex] = extended_float_to_bf16(op(extended_bf16_to_float(input[scalarIndex])));
        }
    }
}

struct RsqrtOp {
    float4 operator()(float4 value) const { return float4(1.0f) / sqrt(value); }
    float operator()(float value) const { return 1.0f / sqrt(value); }
};

struct ExpOp {
    float4 operator()(float4 value) const { return precise::exp(value); }
    float operator()(float value) const { return precise::exp(value); }
};

struct LogOp {
    float4 operator()(float4 value) const { return precise::log(value); }
    float operator()(float value) const { return precise::log(value); }
};

struct SinOp {
    float4 operator()(float4 value) const { return precise::sin(value); }
    float operator()(float value) const { return precise::sin(value); }
};

struct CosOp {
    float4 operator()(float4 value) const { return precise::cos(value); }
    float operator()(float value) const { return precise::cos(value); }
};

struct TanhOp {
    float4 operator()(float4 value) const { return precise::tanh(value); }
    float operator()(float value) const { return precise::tanh(value); }
};

struct GeluOp {
    float4 operator()(float4 value) const {
        return metal_gelu_float4(value);
    }

    float operator()(float value) const {
        return metal_gelu_softfloat_scalar(value);
    }
};

struct SigmoidOp {
    float4 operator()(float4 value) const { return float4(1.0f) / (float4(1.0f) + precise::exp(-value)); }
    float operator()(float value) const { return 1.0f / (1.0f + precise::exp(-value)); }
};

struct SiluOp {
    float4 operator()(float4 value) const { return value / (float4(1.0f) + precise::exp(-value)); }
    float operator()(float value) const { return value / (1.0f + precise::exp(-value)); }
};

struct SoftsignOp {
    float4 operator()(float4 value) const { return value / (float4(1.0f) + fabs(value)); }
    float operator()(float value) const { return value / (1.0f + fabs(value)); }
};

struct ELUOp {
    float4 operator()(float4 value) const {
        return select(precise::exp(value) - float4(1.0f), value, value > float4(0.0f));
    }

    float operator()(float value) const {
        return value > 0.0f ? value : precise::exp(value) - 1.0f;
    }
};

struct SELUOp {
    float4 operator()(float4 value) const {
        float4 negative = metalSELUScale * metalSELUAlpha * (precise::exp(value) - float4(1.0f));
        return select(negative, metalSELUScale * value, value > float4(0.0f));
    }

    float operator()(float value) const {
        if (value > 0.0f) {
            return metalSELUScale * value;
        }

        return metalSELUScale * metalSELUAlpha * (precise::exp(value) - 1.0f);
    }
};

struct LeakyReLUOp {
    float4 operator()(float4 value) const {
        return select(metalLeakyReLUSlope * value, value, value > float4(0.0f));
    }

    float operator()(float value) const {
        return value > 0.0f ? value : metalLeakyReLUSlope * value;
    }
};

struct HardSigmoidOp {
    float4 operator()(float4 value) const {
        return clamp(value / float4(6.0f) + float4(0.5f), float4(0.0f), float4(1.0f));
    }

    float operator()(float value) const {
        return clamp(value / 6.0f + 0.5f, 0.0f, 1.0f);
    }
};

struct HardSwishOp {
    float4 operator()(float4 value) const {
        return value * clamp((value + float4(3.0f)) / float4(6.0f), float4(0.0f), float4(1.0f));
    }

    float operator()(float value) const {
        return value * clamp((value + 3.0f) / 6.0f, 0.0f, 1.0f);
    }
};

constant float metalCeluAlpha = 1.0f;
constant float metalGeluTanhAlpha = 0.7978845608028654f;
constant float metalGeluTanhBeta = 0.044715f;
constant float metalQuickGeluScale = 1.702f;
constant float metalHardTanhMin = -1.0f;
constant float metalHardTanhMax = 1.0f;

struct Log1pOp {
    float4 operator()(float4 value) const { return precise::log(float4(1.0f) + value); }
    float operator()(float value) const { return precise::log(1.0f + value); }
};

struct Expm1Op {
    float4 operator()(float4 value) const { return precise::exp(value) - float4(1.0f); }
    float operator()(float value) const { return precise::exp(value) - 1.0f; }
};

struct CeluOp {
    float4 operator()(float4 value) const {
        return select(metalCeluAlpha * (precise::exp(value / metalCeluAlpha) - float4(1.0f)), value, value > float4(0.0f));
    }
    float operator()(float value) const {
        return value > 0.0f ? value : metalCeluAlpha * (precise::exp(value / metalCeluAlpha) - 1.0f);
    }
};

struct SoftplusOp {
    float4 operator()(float4 value) const {
        return select(value, precise::log(float4(1.0f) + precise::exp(value)), value <= float4(20.0f));
    }
    float operator()(float value) const {
        return value > 20.0f ? value : precise::log(1.0f + precise::exp(value));
    }
};

struct MishOp {
    float4 operator()(float4 value) const {
        float4 softplus = select(value, precise::log(float4(1.0f) + precise::exp(value)), value <= float4(20.0f));
        return value * precise::tanh(softplus);
    }
    float operator()(float value) const {
        float softplus = value > 20.0f ? value : precise::log(1.0f + precise::exp(value));
        return value * precise::tanh(softplus);
    }
};

struct LogSigmoidOp {
    float4 operator()(float4 value) const {
        float4 softplus = select(-value, precise::log(float4(1.0f) + precise::exp(-value)), -value <= float4(20.0f));
        return -softplus;
    }
    float operator()(float value) const {
        float softplus = -value > 20.0f ? -value : precise::log(1.0f + precise::exp(-value));
        return -softplus;
    }
};

struct GeluTanhOp {
    float4 operator()(float4 value) const {
        float4 inner = metalGeluTanhAlpha * (value + metalGeluTanhBeta * value * value * value);
        return float4(0.5f) * value * (float4(1.0f) + precise::tanh(inner));
    }
    float operator()(float value) const {
        float inner = metalGeluTanhAlpha * (value + metalGeluTanhBeta * value * value * value);
        return 0.5f * value * (1.0f + precise::tanh(inner));
    }
};

struct HardTanhOp {
    float4 operator()(float4 value) const {
        return clamp(value, float4(metalHardTanhMin), float4(metalHardTanhMax));
    }
    float operator()(float value) const {
        return clamp(value, metalHardTanhMin, metalHardTanhMax);
    }
};

struct HardGeluOp {
    float4 operator()(float4 value) const {
        float4 inner = value + float4(3.0f);
        inner = clamp(inner, float4(0.0f), float4(6.0f));
        return value * (inner / float4(6.0f));
    }
    float operator()(float value) const {
        float inner = value + 3.0f;
        if (inner < 0.0f) {
            inner = 0.0f;
        }
        if (inner > 6.0f) {
            inner = 6.0f;
        }
        return value * (inner / 6.0f);
    }
};

struct QuickGeluOp {
    float4 operator()(float4 value) const {
        return value / (float4(1.0f) + precise::exp(-metalQuickGeluScale * value));
    }
    float operator()(float value) const {
        return value / (1.0f + precise::exp(-metalQuickGeluScale * value));
    }
};

struct TanhShrinkOp {
    float4 operator()(float4 value) const { return value - precise::tanh(value); }
    float operator()(float value) const { return value - precise::tanh(value); }
};

#define EXTENDED_UNARY_FLOAT32_KERNEL(name, op) \
kernel void name##_float32( \
    device const float4* inputVector [[buffer(0)]], \
    device float4* outVector [[buffer(1)]], \
    constant uint& count [[buffer(2)]], \
    uint index [[thread_position_in_grid]] \
) { \
    extended_unary_float32(inputVector, outVector, count, index, op{}); \
}

#define EXTENDED_UNARY_FLOAT16_KERNEL(name, op) \
kernel void name##_float16( \
    device const half4* inputVector [[buffer(0)]], \
    device half4* outVector [[buffer(1)]], \
    constant uint& count [[buffer(2)]], \
    uint index [[thread_position_in_grid]] \
) { \
    extended_unary_float16(inputVector, outVector, count, index, op{}); \
}

#define EXTENDED_UNARY_BFLOAT16_KERNEL(name, op) \
kernel void name##_bfloat16( \
    device const ushort4* inputVector [[buffer(0)]], \
    device ushort4* outVector [[buffer(1)]], \
    constant uint& count [[buffer(2)]], \
    uint index [[thread_position_in_grid]] \
) { \
    extended_unary_bfloat16(inputVector, outVector, count, index, op{}); \
}

#define EXTENDED_UNARY_KERNELS(dtype_macro) \
dtype_macro(rsqrt, RsqrtOp) \
dtype_macro(exp, ExpOp) \
dtype_macro(log, LogOp) \
dtype_macro(sin, SinOp) \
dtype_macro(cos, CosOp) \
dtype_macro(tanh, TanhOp) \
dtype_macro(gelu, GeluOp) \
dtype_macro(sigmoid, SigmoidOp) \
dtype_macro(silu, SiluOp) \
dtype_macro(swish, SiluOp) \
dtype_macro(softsign, SoftsignOp) \
dtype_macro(elu, ELUOp) \
dtype_macro(selu, SELUOp) \
dtype_macro(leaky_relu, LeakyReLUOp) \
dtype_macro(hardsigmoid, HardSigmoidOp) \
dtype_macro(hardswish, HardSwishOp) \
dtype_macro(log1p, Log1pOp) \
dtype_macro(expm1, Expm1Op) \
dtype_macro(celu, CeluOp) \
dtype_macro(softplus, SoftplusOp) \
dtype_macro(mish, MishOp) \
dtype_macro(log_sigmoid, LogSigmoidOp) \
dtype_macro(gelu_tanh, GeluTanhOp) \
dtype_macro(hardtanh, HardTanhOp) \
dtype_macro(hard_gelu, HardGeluOp) \
dtype_macro(quick_gelu, QuickGeluOp) \
dtype_macro(tanh_shrink, TanhShrinkOp)

EXTENDED_UNARY_KERNELS(EXTENDED_UNARY_FLOAT32_KERNEL)
EXTENDED_UNARY_KERNELS(EXTENDED_UNARY_FLOAT16_KERNEL)
EXTENDED_UNARY_KERNELS(EXTENDED_UNARY_BFLOAT16_KERNEL)
