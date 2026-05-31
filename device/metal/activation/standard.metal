#include <metal_stdlib>
#include "activation.metal"
#include "../elementwise/elementwise_f64_math.metalinc"
#include "activation_ops_f16.metalinc"
#include "activation_ops_bf16.metalinc"

using namespace metal;

static inline float metal_gelu_tanh_softfloat_scalar(float value) {
    float valueCubed = value * value * value;
    float inner = metalGeluTanhAlpha * fma(metalGeluTanhBeta, valueCubed, value);
    float tanhValue = metal_fast_tanh_rational(inner);
    ulong value64 = metal_sf32_to64(as_type<uint>(value));
    ulong tanh64 = metal_sf32_to64(as_type<uint>(tanhValue));
    ulong onePlusTanh = metal_sf64_add(SF64_ONE, tanh64);
    ulong product = metal_sf64_mul(SF64_HALF, metal_sf64_mul(value64, onePlusTanh));

    return as_type<float>(metal_sf64_to32(product));
}


struct ReluOp {
    float4 operator()(float4 value) const { return max(float4(0.0f), value); }
    float operator()(float value) const { return value > 0.0f ? value : 0.0f; }
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
        float4 negative = metalActivationSELUScale * metalActivationSELUAlpha * (precise::exp(value) - float4(1.0f));
        return select(negative, metalActivationSELUScale * value, value > float4(0.0f));
    }

    float operator()(float value) const {
        if (value > 0.0f) {
            return metalActivationSELUScale * value;
        }

        return metalActivationSELUScale * metalActivationSELUAlpha * (precise::exp(value) - 1.0f);
    }
};

struct LeakyReLUOp {
    float4 operator()(float4 value) const {
        return select(metalActivationLeakyReLUSlope * value, value, value > float4(0.0f));
    }

    float operator()(float value) const {
        return value > 0.0f ? value : metalActivationLeakyReLUSlope * value;
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
        return float4(
            metal_gelu_tanh_softfloat_scalar(value.x),
            metal_gelu_tanh_softfloat_scalar(value.y),
            metal_gelu_tanh_softfloat_scalar(value.z),
            metal_gelu_tanh_softfloat_scalar(value.w)
        );
    }

    float operator()(float value) const {
        return metal_gelu_tanh_softfloat_scalar(value);
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

#define STANDARD_UNARY_FLOAT32_KERNEL(name, op) \
kernel void name##_float32( \
    device const float4* inputVector [[buffer(0)]], \
    device float4* outVector [[buffer(1)]], \
    constant uint& count [[buffer(2)]], \
    uint index [[thread_position_in_grid]] \
) { \
    activation_unary_float32(inputVector, outVector, count, index, op{}); \
}

#define STANDARD_UNARY_FLOAT16_KERNEL(name, op) \
kernel void name##_float16( \
    device const half4* inputVector [[buffer(0)]], \
    device half4* outVector [[buffer(1)]], \
    constant uint& count [[buffer(2)]], \
    uint index [[thread_position_in_grid]] \
) { \
    activation_unary_float16(inputVector, outVector, count, index, op{}); \
}

#define STANDARD_UNARY_BFLOAT16_KERNEL(name, op) \
kernel void name##_bfloat16( \
    device const ushort4* inputVector [[buffer(0)]], \
    device ushort4* outVector [[buffer(1)]], \
    constant uint& count [[buffer(2)]], \
    uint index [[thread_position_in_grid]] \
) { \
    activation_unary_bfloat16(inputVector, outVector, count, index, op{}); \
}

#define STANDARD_UNARY_KERNELS(dtype_macro) \
dtype_macro(exp, ExpOp) \
dtype_macro(log, LogOp) \
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
dtype_macro(tanh_shrink, TanhShrinkOp) \
dtype_macro(relu, ReluOp) \

STANDARD_UNARY_KERNELS(STANDARD_UNARY_FLOAT32_KERNEL)

#define STANDARD_HALF_UNARY_KERNELS(dtype_macro) \
dtype_macro(exp, HalfExpOp) \
dtype_macro(log, HalfLogOp) \
dtype_macro(tanh, HalfTanhOp) \
dtype_macro(gelu, HalfGeluOp) \
dtype_macro(sigmoid, HalfSigmoidOp) \
dtype_macro(silu, HalfSiluOp) \
dtype_macro(swish, HalfSiluOp) \
dtype_macro(softsign, HalfSoftsignOp) \
dtype_macro(elu, HalfELUOp) \
dtype_macro(selu, HalfSELUOp) \
dtype_macro(leaky_relu, HalfLeakyReLUOp) \
dtype_macro(hardsigmoid, HalfHardSigmoidOp) \
dtype_macro(hardswish, HalfHardSwishOp) \
dtype_macro(log1p, HalfLog1pOp) \
dtype_macro(expm1, HalfExpm1Op) \
dtype_macro(celu, HalfCeluOp) \
dtype_macro(softplus, HalfSoftplusOp) \
dtype_macro(mish, HalfMishOp) \
dtype_macro(log_sigmoid, HalfLogSigmoidOp) \
dtype_macro(gelu_tanh, HalfGeluTanhOp) \
dtype_macro(hardtanh, HalfHardTanhOp) \
dtype_macro(hard_gelu, HalfHardGeluOp) \
dtype_macro(quick_gelu, HalfQuickGeluOp) \
dtype_macro(tanh_shrink, HalfTanhShrinkOp) \
dtype_macro(relu, HalfReluOp)

#define STANDARD_BF16_UNARY_KERNELS(dtype_macro) \
dtype_macro(exp, BF16ExpOp) \
dtype_macro(log, BF16LogOp) \
dtype_macro(tanh, BF16TanhOp) \
dtype_macro(gelu, BF16GeluOp) \
dtype_macro(sigmoid, BF16SigmoidOp) \
dtype_macro(silu, BF16SiluOp) \
dtype_macro(swish, BF16SiluOp) \
dtype_macro(softsign, BF16SoftsignOp) \
dtype_macro(elu, BF16ELUOp) \
dtype_macro(selu, BF16SELUOp) \
dtype_macro(leaky_relu, BF16LeakyReLUOp) \
dtype_macro(hardsigmoid, BF16HardSigmoidOp) \
dtype_macro(hardswish, BF16HardSwishOp) \
dtype_macro(log1p, BF16Log1pOp) \
dtype_macro(expm1, BF16Expm1Op) \
dtype_macro(celu, BF16CeluOp) \
dtype_macro(softplus, BF16SoftplusOp) \
dtype_macro(mish, BF16MishOp) \
dtype_macro(log_sigmoid, BF16LogSigmoidOp) \
dtype_macro(gelu_tanh, BF16GeluTanhOp) \
dtype_macro(hardtanh, BF16HardTanhOp) \
dtype_macro(hard_gelu, BF16HardGeluOp) \
dtype_macro(quick_gelu, BF16QuickGeluOp) \
dtype_macro(tanh_shrink, BF16TanhShrinkOp) \
dtype_macro(relu, BF16ReluOp)

STANDARD_HALF_UNARY_KERNELS(STANDARD_UNARY_FLOAT16_KERNEL)
STANDARD_BF16_UNARY_KERNELS(STANDARD_UNARY_BFLOAT16_KERNEL)
