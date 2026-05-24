#ifndef PUTER_DEVICE_METAL_ACTIVATION_ACTIVATION_METAL
#define PUTER_DEVICE_METAL_ACTIVATION_ACTIVATION_METAL

#include <metal_stdlib>

using namespace metal;

constant float metalActivationSELUAlpha = 1.67326324235437728482f;
constant float metalActivationSELUScale = 1.05070098735548049342f;
constant float metalActivationLeakyReLUSlope = 0.01f;

static inline float metal_fast_tanh(float value) {
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

static inline float metal_fast_gelu_tanh(float value) {
    double valueFloat64 = double(value);
    double inner = 0.7978845608028654 * (
        valueFloat64 + 0.044715 * valueFloat64 * valueFloat64 * valueFloat64
    );

    return float(0.5 * valueFloat64 * (1.0 + double(metal_fast_tanh(float(inner)))));
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
