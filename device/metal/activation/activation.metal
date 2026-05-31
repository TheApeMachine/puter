#ifndef PUTER_DEVICE_METAL_ACTIVATION_ACTIVATION_METAL
#define PUTER_DEVICE_METAL_ACTIVATION_ACTIVATION_METAL

#include <metal_stdlib>

using namespace metal;

// Referenced by selu / leaky_relu kernels in standard.metal via include;
// standalone compilation of activation.metal sees no caller and Clang
// would warn -Wunused-const-variable without the attribute.
__attribute__((unused))
constant float metalActivationSELUAlpha = 1.67326324235437728482f;
__attribute__((unused))
constant float metalActivationSELUScale = 1.05070098735548049342f;
__attribute__((unused))
constant float metalActivationLeakyReLUSlope = 0.01f;

constant float metalGeluTanhAlpha = 0.7978845608028654f;
constant float metalGeluTanhBeta = 0.044715f;
constant float metalFastExp32Log2E = 1.4426950408889634f;
constant float metalFastExp32Ln2 = 0.6931471805599453f;
constant float metalFastExp32Min = -87.33654f;
constant float metalFastExp32Max = 88.72283f;
constant float metalFastExp32Overflow = 0x1.fffffep127f;
constant float metalFastExp32PolyC7 = 0.00019841270f;
constant float metalFastExp32PolyC6 = 0.0013888889f;
constant float metalFastExp32PolyC5 = 0.008333334f;
constant float metalFastExp32PolyC4 = 0.041666667f;
constant float metalFastExp32PolyC3 = 0.16666667f;
constant float metalFastExp32PolyC2 = 0.5f;
constant float metalFastExp32PolyC1 = 1.0f;
constant float metalFastExp32PolyC0 = 1.0f;

static inline float metal_exp32_horner(float value) {
    float scaled = value * metalFastExp32Log2E;
    float roundedK = rint(scaled);
    float fraction = value - roundedK * metalFastExp32Ln2;
    float poly = metalFastExp32PolyC7;

    poly = fma(fraction, poly, metalFastExp32PolyC6);
    poly = fma(fraction, poly, metalFastExp32PolyC5);
    poly = fma(fraction, poly, metalFastExp32PolyC4);
    poly = fma(fraction, poly, metalFastExp32PolyC3);
    poly = fma(fraction, poly, metalFastExp32PolyC2);
    poly = fma(fraction, poly, metalFastExp32PolyC1);
    poly = fma(fraction, poly, metalFastExp32PolyC0);

    int32_t exponentInt = int32_t(roundedK);
    uint scaleBits = as_type<uint>(exponentInt + 127) << 23;

    return poly * as_type<float>(scaleBits);
}

static inline float metal_fast_exp32(float value) {
    if (value < metalFastExp32Min) {
        return 0.0f;
    }

    if (value > metalFastExp32Max) {
        return metalFastExp32Overflow;
    }

    return metal_exp32_horner(value);
}

static inline float metal_fast_tanh_exp32(float value) {
    float expTwoValue = metal_fast_exp32(2.0f * value);

    return (expTwoValue - 1.0f) / (expTwoValue + 1.0f);
}

// Matches cpumath.FastTanh32 in device/cpu/math/f32.go.
static inline float metal_fast_tanh_rational(float value) {
    if (value > 4.92f) {
        return 1.0f;
    }

    if (value < -4.92f) {
        return -1.0f;
    }

    float valueSquared = value * value;
    float numerator = value * (
        135135.0f + valueSquared * (17325.0f + valueSquared * (378.0f + valueSquared))
    );
    float denominator = 135135.0f + valueSquared * (
        62370.0f + valueSquared * (3150.0f + valueSquared * 28.0f)
    );

    return numerator / denominator;
}

// Called from standard.metal's gelu_tanh kernels via `#include
// "activation.metal"`. Standalone compilation of activation.metal sees
// no caller, so silence -Wunused-function with __attribute__((unused)).
__attribute__((unused))
static inline float metal_fast_gelu_tanh(float value) {
    float valueCubed = value * value * value;
    float inner = metalGeluTanhAlpha * fma(metalGeluTanhBeta, valueCubed, value);
    float tanhValue = metal_fast_tanh_rational(inner);

#pragma METAL fp_contract(off)
    return 0.5f * value * (1.0f + tanhValue);
#pragma METAL fp_contract(on)
}

template <typename UnaryOp>
static inline void activation_unary_float32(
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
static inline void activation_unary_float16(
    device const half4* inputVector,
    device half4* outVector,
    constant uint& count,
    uint index [[thread_position_in_grid]],
    UnaryOp op
) {
    uint base = index * 4;

    if (base + 3 < count) {
        outVector[index] = op(inputVector[index]);
        return;
    }

    device const half* input = reinterpret_cast<device const half*>(inputVector);
    device half* out = reinterpret_cast<device half*>(outVector);

    for (uint offset = 0; offset < 4; offset++) {
        uint scalarIndex = base + offset;

        if (scalarIndex < count) {
            out[scalarIndex] = op(input[scalarIndex]);
        }
    }
}

template <typename UnaryOp>
static inline void activation_unary_bfloat16(
    device const ushort4* inputVector,
    device ushort4* outVector,
    constant uint& count,
    uint index [[thread_position_in_grid]],
    UnaryOp op
) {
    uint base = index * 4;

    if (base + 3 < count) {
        bfloat4 loaded = as_type<bfloat4>(inputVector[index]);
        outVector[index] = as_type<ushort4>(op(loaded));
        return;
    }

    device const ushort* input = reinterpret_cast<device const ushort*>(inputVector);
    device ushort* out = reinterpret_cast<device ushort*>(outVector);

    for (uint offset = 0; offset < 4; offset++) {
        uint scalarIndex = base + offset;

        if (scalarIndex < count) {
            bfloat loaded = as_type<bfloat>(input[scalarIndex]);
            out[scalarIndex] = as_type<ushort>(op(loaded));
        }
    }
}

#endif
