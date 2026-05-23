// --- elementwise_extended.metal ---
#include <metal_stdlib>

using namespace metal;

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

struct SinOp {
    float4 operator()(float4 value) const { return precise::sin(value); }
    float operator()(float value) const { return precise::sin(value); }
};

struct CosOp {
    float4 operator()(float4 value) const { return precise::cos(value); }
    float operator()(float value) const { return precise::cos(value); }
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
dtype_macro(sin, SinOp) \
dtype_macro(cos, CosOp)

EXTENDED_UNARY_KERNELS(EXTENDED_UNARY_FLOAT32_KERNEL)
EXTENDED_UNARY_KERNELS(EXTENDED_UNARY_FLOAT16_KERNEL)
EXTENDED_UNARY_KERNELS(EXTENDED_UNARY_BFLOAT16_KERNEL)
