#ifndef PUTER_DEVICE_METAL_ACTIVATION_ACTIVATION_METAL
#define PUTER_DEVICE_METAL_ACTIVATION_ACTIVATION_METAL

#include <metal_stdlib>

using namespace metal;

constant float metalActivationSELUAlpha = 1.67326324235437728482f;
constant float metalActivationSELUScale = 1.05070098735548049342f;
constant float metalActivationLeakyReLUSlope = 0.01f;

static inline float activation_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort activation_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

static inline float4 activation_bf16_to_float4(ushort4 value) {
    return float4(
        activation_bf16_to_float(value.x),
        activation_bf16_to_float(value.y),
        activation_bf16_to_float(value.z),
        activation_bf16_to_float(value.w)
    );
}

static inline ushort4 activation_float4_to_bf16(float4 value) {
    return ushort4(
        activation_float_to_bf16(value.x),
        activation_float_to_bf16(value.y),
        activation_float_to_bf16(value.z),
        activation_float_to_bf16(value.w)
    );
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
static inline void activation_unary_bfloat16(
    device const ushort4* inputVector,
    device ushort4* outVector,
    constant uint& count,
    uint index [[thread_position_in_grid]],
    UnaryOp op
) {
    uint base = index * 4;

    if (base + 3 < count) {
        outVector[index] = activation_float4_to_bf16(op(activation_bf16_to_float4(inputVector[index])));
        return;
    }

    device const ushort* input = reinterpret_cast<device const ushort*>(inputVector);
    device ushort* out = reinterpret_cast<device ushort*>(outVector);

    for (uint offset = 0; offset < 4; offset++) {
        uint scalarIndex = base + offset;

        if (scalarIndex < count) {
            out[scalarIndex] = activation_float_to_bf16(op(activation_bf16_to_float(input[scalarIndex])));
        }
    }
}

#endif
