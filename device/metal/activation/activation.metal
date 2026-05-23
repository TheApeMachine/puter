#ifndef PUTER_DEVICE_METAL_ACTIVATION_ACTIVATION_METAL
#define PUTER_DEVICE_METAL_ACTIVATION_ACTIVATION_METAL

#include <metal_stdlib>

using namespace metal;

constant float metalActivationSELUAlpha = 1.67326324235437728482f;
constant float metalActivationSELUScale = 1.05070098735548049342f;
constant float metalActivationLeakyReLUSlope = 0.01f;

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
