// --- activation_geglu.metal ---
#include <metal_stdlib>
#include "elementwise_gelu_f64.metalinc"

using namespace metal;

// gate * FastGelu32(up) — matches pkg/backend/device/cpu/math.FastGeGLU32.
static inline float4 geglu_float4(float4 gate, float4 up) {
    return gate * metal_gelu_float4(up);
}

static inline float geglu_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort geglu_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

static inline void geglu_write_tail(
    device float* destination,
    uint offset,
    uint tail,
    float4 result
) {
    for (uint lane = 0; lane < tail; ++lane) {
        destination[offset + lane] = result[lane];
    }
}

static inline void geglu_float32_kernel(
    device float4* destination [[buffer(0)]],
    device const float4* gateVector [[buffer(1)]],
    device const float4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    device float* destinationScalar = (device float*)destination;
    device const float* gateScalar = (device const float*)gateVector;
    device const float* upScalar = (device const float*)upVector;
    uint offset = index * 4u;

    if (offset + 4u <= count) {
        destination[index] = geglu_float4(gateVector[index], upVector[index]);
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    float4 gateVec = float4(
        gateScalar[offset],
        gateScalar[offset + 1u],
        gateScalar[offset + 2u],
        gateScalar[offset + 3u]
    );
    float4 upVec = float4(
        upScalar[offset],
        upScalar[offset + 1u],
        upScalar[offset + 2u],
        upScalar[offset + 3u]
    );
    float4 result = geglu_float4(gateVec, upVec);

    geglu_write_tail(destinationScalar, offset, tail, result);
}

kernel void geglu_float32(
    device float4* destination [[buffer(0)]],
    device const float4* gateVector [[buffer(1)]],
    device const float4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    geglu_float32_kernel(destination, gateVector, upVector, count, index);
}

static inline void geglu_float16_kernel(
    device half4* destination [[buffer(0)]],
    device const half4* gateVector [[buffer(1)]],
    device const half4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    device half* destinationScalar = (device half*)destination;
    device const half* gateScalar = (device const half*)gateVector;
    device const half* upScalar = (device const half*)upVector;
    uint offset = index * 4u;

    if (offset + 4u <= count) {
        float4 gateVec = float4(gateVector[index]);
        float4 upVec = float4(upVector[index]);
        destination[index] = half4(geglu_float4(gateVec, upVec));
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    float4 gateVec = float4(
        float(gateScalar[offset]),
        float(gateScalar[offset + 1u]),
        float(gateScalar[offset + 2u]),
        float(gateScalar[offset + 3u])
    );
    float4 upVec = float4(
        float(upScalar[offset]),
        float(upScalar[offset + 1u]),
        float(upScalar[offset + 2u]),
        float(upScalar[offset + 3u])
    );
    float4 result = geglu_float4(gateVec, upVec);

    for (uint lane = 0; lane < tail; ++lane) {
        destinationScalar[offset + lane] = half(result[lane]);
    }
}

kernel void geglu_float16(
    device half4* destination [[buffer(0)]],
    device const half4* gateVector [[buffer(1)]],
    device const half4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    geglu_float16_kernel(destination, gateVector, upVector, count, index);
}

static inline void geglu_bfloat16_kernel(
    device ushort4* destination [[buffer(0)]],
    device const ushort4* gateVector [[buffer(1)]],
    device const ushort4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    device ushort* destinationScalar = (device ushort*)destination;
    device const ushort* gateScalar = (device const ushort*)gateVector;
    device const ushort* upScalar = (device const ushort*)upVector;
    uint offset = index * 4u;

    if (offset + 4u <= count) {
        float4 gateVec = float4(
            geglu_bf16_to_float(gateVector[index].x),
            geglu_bf16_to_float(gateVector[index].y),
            geglu_bf16_to_float(gateVector[index].z),
            geglu_bf16_to_float(gateVector[index].w)
        );
        float4 upVec = float4(
            geglu_bf16_to_float(upVector[index].x),
            geglu_bf16_to_float(upVector[index].y),
            geglu_bf16_to_float(upVector[index].z),
            geglu_bf16_to_float(upVector[index].w)
        );
        float4 result = geglu_float4(gateVec, upVec);
        destination[index] = ushort4(
            geglu_float_to_bf16(result.x),
            geglu_float_to_bf16(result.y),
            geglu_float_to_bf16(result.z),
            geglu_float_to_bf16(result.w)
        );
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    float4 gateVec = float4(
        geglu_bf16_to_float(gateScalar[offset]),
        geglu_bf16_to_float(gateScalar[offset + 1u]),
        geglu_bf16_to_float(gateScalar[offset + 2u]),
        geglu_bf16_to_float(gateScalar[offset + 3u])
    );
    float4 upVec = float4(
        geglu_bf16_to_float(upScalar[offset]),
        geglu_bf16_to_float(upScalar[offset + 1u]),
        geglu_bf16_to_float(upScalar[offset + 2u]),
        geglu_bf16_to_float(upScalar[offset + 3u])
    );
    float4 result = geglu_float4(gateVec, upVec);

    for (uint lane = 0; lane < tail; ++lane) {
        destinationScalar[offset + lane] = geglu_float_to_bf16(result[lane]);
    }
}

kernel void geglu_bfloat16(
    device ushort4* destination [[buffer(0)]],
    device const ushort4* gateVector [[buffer(1)]],
    device const ushort4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    geglu_bfloat16_kernel(destination, gateVector, upVector, count, index);
}

// --- activation_geglu_tanh.metal ---
#include <metal_stdlib>

using namespace metal;

// pkg/backend/device/cpu/math/constant.go GeluTanhAlpha, GeluTanhBeta.
constant float gegluTanhAlpha = 0.7978845608028654f;
constant float gegluTanhBeta = 0.044715f;

// Padé tanh — matches pkg/backend/device/cpu/math.FastTanh32.
static inline float metal_fast_tanh32(float value) {
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

// tanh GELU — matches pkg/backend/device/cpu/math.FastGeluTanh32 (per-lane f32).
static inline float metal_fast_gelu_tanh32(float value) {
    float cubic = value * value * value;
    float inner = gegluTanhAlpha * (value + gegluTanhBeta * cubic);

    return 0.5f * value * (1.0f + metal_fast_tanh32(inner));
}

static inline float4 metal_fast_gelu_tanh32_float4(float4 value) {
    return float4(
        metal_fast_gelu_tanh32(value.x),
        metal_fast_gelu_tanh32(value.y),
        metal_fast_gelu_tanh32(value.z),
        metal_fast_gelu_tanh32(value.w)
    );
}

// gate * FastGeluTanh32(up) — matches pkg/backend/device/cpu/math.FastGeGLUTanh32.
static inline float4 geglu_tanh_float4(float4 gate, float4 up) {
    return gate * metal_fast_gelu_tanh32_float4(up);
}

static inline float geglu_tanh_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort geglu_tanh_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

static inline void geglu_tanh_write_tail(
    device float* destination,
    uint offset,
    uint tail,
    float4 result
) {
    for (uint lane = 0; lane < tail; ++lane) {
        destination[offset + lane] = result[lane];
    }
}

static inline void geglu_tanh_float32_kernel(
    device float4* destination [[buffer(0)]],
    device const float4* gateVector [[buffer(1)]],
    device const float4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    device float* destinationScalar = (device float*)destination;
    device const float* gateScalar = (device const float*)gateVector;
    device const float* upScalar = (device const float*)upVector;
    uint offset = index * 4u;

    if (offset + 4u <= count) {
        destination[index] = geglu_tanh_float4(gateVector[index], upVector[index]);
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    float4 gateVec = float4(
        gateScalar[offset],
        gateScalar[offset + 1u],
        gateScalar[offset + 2u],
        gateScalar[offset + 3u]
    );
    float4 upVec = float4(
        upScalar[offset],
        upScalar[offset + 1u],
        upScalar[offset + 2u],
        upScalar[offset + 3u]
    );
    float4 result = geglu_tanh_float4(gateVec, upVec);

    geglu_tanh_write_tail(destinationScalar, offset, tail, result);
}

kernel void geglu_tanh_float32(
    device float4* destination [[buffer(0)]],
    device const float4* gateVector [[buffer(1)]],
    device const float4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    geglu_tanh_float32_kernel(destination, gateVector, upVector, count, index);
}

static inline void geglu_tanh_float16_kernel(
    device half4* destination [[buffer(0)]],
    device const half4* gateVector [[buffer(1)]],
    device const half4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    device half* destinationScalar = (device half*)destination;
    device const half* gateScalar = (device const half*)gateVector;
    device const half* upScalar = (device const half*)upVector;
    uint offset = index * 4u;

    if (offset + 4u <= count) {
        float4 gateVec = float4(gateVector[index]);
        float4 upVec = float4(upVector[index]);
        destination[index] = half4(geglu_tanh_float4(gateVec, upVec));
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    float4 gateVec = float4(
        float(gateScalar[offset]),
        float(gateScalar[offset + 1u]),
        float(gateScalar[offset + 2u]),
        float(gateScalar[offset + 3u])
    );
    float4 upVec = float4(
        float(upScalar[offset]),
        float(upScalar[offset + 1u]),
        float(upScalar[offset + 2u]),
        float(upScalar[offset + 3u])
    );
    float4 result = geglu_tanh_float4(gateVec, upVec);

    for (uint lane = 0; lane < tail; ++lane) {
        destinationScalar[offset + lane] = half(result[lane]);
    }
}

kernel void geglu_tanh_float16(
    device half4* destination [[buffer(0)]],
    device const half4* gateVector [[buffer(1)]],
    device const half4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    geglu_tanh_float16_kernel(destination, gateVector, upVector, count, index);
}

static inline void geglu_tanh_bfloat16_kernel(
    device ushort4* destination [[buffer(0)]],
    device const ushort4* gateVector [[buffer(1)]],
    device const ushort4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    device ushort* destinationScalar = (device ushort*)destination;
    device const ushort* gateScalar = (device const ushort*)gateVector;
    device const ushort* upScalar = (device const ushort*)upVector;
    uint offset = index * 4u;

    if (offset + 4u <= count) {
        float4 gateVec = float4(
            geglu_tanh_bf16_to_float(gateVector[index].x),
            geglu_tanh_bf16_to_float(gateVector[index].y),
            geglu_tanh_bf16_to_float(gateVector[index].z),
            geglu_tanh_bf16_to_float(gateVector[index].w)
        );
        float4 upVec = float4(
            geglu_tanh_bf16_to_float(upVector[index].x),
            geglu_tanh_bf16_to_float(upVector[index].y),
            geglu_tanh_bf16_to_float(upVector[index].z),
            geglu_tanh_bf16_to_float(upVector[index].w)
        );
        float4 result = geglu_tanh_float4(gateVec, upVec);
        destination[index] = ushort4(
            geglu_tanh_float_to_bf16(result.x),
            geglu_tanh_float_to_bf16(result.y),
            geglu_tanh_float_to_bf16(result.z),
            geglu_tanh_float_to_bf16(result.w)
        );
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    float4 gateVec = float4(
        geglu_tanh_bf16_to_float(gateScalar[offset]),
        geglu_tanh_bf16_to_float(gateScalar[offset + 1u]),
        geglu_tanh_bf16_to_float(gateScalar[offset + 2u]),
        geglu_tanh_bf16_to_float(gateScalar[offset + 3u])
    );
    float4 upVec = float4(
        geglu_tanh_bf16_to_float(upScalar[offset]),
        geglu_tanh_bf16_to_float(upScalar[offset + 1u]),
        geglu_tanh_bf16_to_float(upScalar[offset + 2u]),
        geglu_tanh_bf16_to_float(upScalar[offset + 3u])
    );
    float4 result = geglu_tanh_float4(gateVec, upVec);

    for (uint lane = 0; lane < tail; ++lane) {
        destinationScalar[offset + lane] = geglu_tanh_float_to_bf16(result[lane]);
    }
}

kernel void geglu_tanh_bfloat16(
    device ushort4* destination [[buffer(0)]],
    device const ushort4* gateVector [[buffer(1)]],
    device const ushort4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    geglu_tanh_bfloat16_kernel(destination, gateVector, upVector, count, index);
}

// --- activation_glu.metal ---
#include <metal_stdlib>

using namespace metal;

// Matches pkg/backend/device/cpu/math.FastExp32 (platform scalar for SwiGLU).
static inline float metal_fast_exp32(float value) {
    if (value < -87.33654f) {
        return 0.0f;
    }

    if (value > 88.72283f) {
        return 0x1.fffffep127f;
    }

    float z = value * 1.4426950408889634f;
    int exponentK = int(z);

    if (z < 0.0f) {
        exponentK--;
    }

    float fraction = z - float(exponentK);
    float poly = 1.0f + fraction * (
        0.69314718f + fraction * (
            0.24022650f + fraction * (
                0.05550410f + fraction * (
                    0.00961812f + fraction * 0.00133389f
                )
            )
        )
    );

    uint bits = as_type<uint>(poly) + uint(exponentK) * 8388608u;
    return as_type<float>(bits);
}

// Matches pkg/backend/device/cpu/math.FastSigmoid32 (platform scalar for GLU).
static inline float metal_fast_sigmoid32(float value) {
    if (value >= 0.0f) {
        return 1.0f / (1.0f + metal_fast_exp32(-value));
    }

    float expValue = metal_fast_exp32(value);
    return expValue / (1.0f + expValue);
}

static inline float4 metal_fast_sigmoid32_float4(float4 value) {
    return float4(
        metal_fast_sigmoid32(value.x),
        metal_fast_sigmoid32(value.y),
        metal_fast_sigmoid32(value.z),
        metal_fast_sigmoid32(value.w)
    );
}

// gate * FastSigmoid32(up) — matches pkg/backend/device/cpu/math.FastGLU32.
static inline float4 glu_float4(float4 gate, float4 up) {
    return gate * metal_fast_sigmoid32_float4(up);
}

static inline float glu_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort glu_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

static inline void glu_write_tail(
    device float* destination,
    uint offset,
    uint tail,
    float4 result
) {
    for (uint lane = 0; lane < tail; ++lane) {
        destination[offset + lane] = result[lane];
    }
}

static inline void glu_float32_kernel(
    device float4* destination [[buffer(0)]],
    device const float4* gateVector [[buffer(1)]],
    device const float4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    device float* destinationScalar = (device float*)destination;
    device const float* gateScalar = (device const float*)gateVector;
    device const float* upScalar = (device const float*)upVector;
    uint offset = index * 4u;

    if (offset + 4u <= count) {
        destination[index] = glu_float4(gateVector[index], upVector[index]);
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    float4 gateVec = float4(
        gateScalar[offset],
        gateScalar[offset + 1u],
        gateScalar[offset + 2u],
        gateScalar[offset + 3u]
    );
    float4 upVec = float4(
        upScalar[offset],
        upScalar[offset + 1u],
        upScalar[offset + 2u],
        upScalar[offset + 3u]
    );
    float4 result = glu_float4(gateVec, upVec);

    glu_write_tail(destinationScalar, offset, tail, result);
}

kernel void glu_float32(
    device float4* destination [[buffer(0)]],
    device const float4* gateVector [[buffer(1)]],
    device const float4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    glu_float32_kernel(destination, gateVector, upVector, count, index);
}

static inline void glu_float16_kernel(
    device half4* destination [[buffer(0)]],
    device const half4* gateVector [[buffer(1)]],
    device const half4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    device half* destinationScalar = (device half*)destination;
    device const half* gateScalar = (device const half*)gateVector;
    device const half* upScalar = (device const half*)upVector;
    uint offset = index * 4u;

    if (offset + 4u <= count) {
        float4 gateVec = float4(gateVector[index]);
        float4 upVec = float4(upVector[index]);
        destination[index] = half4(glu_float4(gateVec, upVec));
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    float4 gateVec = float4(
        float(gateScalar[offset]),
        float(gateScalar[offset + 1u]),
        float(gateScalar[offset + 2u]),
        float(gateScalar[offset + 3u])
    );
    float4 upVec = float4(
        float(upScalar[offset]),
        float(upScalar[offset + 1u]),
        float(upScalar[offset + 2u]),
        float(upScalar[offset + 3u])
    );
    float4 result = glu_float4(gateVec, upVec);

    for (uint lane = 0; lane < tail; ++lane) {
        destinationScalar[offset + lane] = half(result[lane]);
    }
}

kernel void glu_float16(
    device half4* destination [[buffer(0)]],
    device const half4* gateVector [[buffer(1)]],
    device const half4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    glu_float16_kernel(destination, gateVector, upVector, count, index);
}

static inline void glu_bfloat16_kernel(
    device ushort4* destination [[buffer(0)]],
    device const ushort4* gateVector [[buffer(1)]],
    device const ushort4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    device ushort* destinationScalar = (device ushort*)destination;
    device const ushort* gateScalar = (device const ushort*)gateVector;
    device const ushort* upScalar = (device const ushort*)upVector;
    uint offset = index * 4u;

    if (offset + 4u <= count) {
        float4 gateVec = float4(
            glu_bf16_to_float(gateVector[index].x),
            glu_bf16_to_float(gateVector[index].y),
            glu_bf16_to_float(gateVector[index].z),
            glu_bf16_to_float(gateVector[index].w)
        );
        float4 upVec = float4(
            glu_bf16_to_float(upVector[index].x),
            glu_bf16_to_float(upVector[index].y),
            glu_bf16_to_float(upVector[index].z),
            glu_bf16_to_float(upVector[index].w)
        );
        float4 result = glu_float4(gateVec, upVec);
        destination[index] = ushort4(
            glu_float_to_bf16(result.x),
            glu_float_to_bf16(result.y),
            glu_float_to_bf16(result.z),
            glu_float_to_bf16(result.w)
        );
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    float4 gateVec = float4(
        glu_bf16_to_float(gateScalar[offset]),
        glu_bf16_to_float(gateScalar[offset + 1u]),
        glu_bf16_to_float(gateScalar[offset + 2u]),
        glu_bf16_to_float(gateScalar[offset + 3u])
    );
    float4 upVec = float4(
        glu_bf16_to_float(upScalar[offset]),
        glu_bf16_to_float(upScalar[offset + 1u]),
        glu_bf16_to_float(upScalar[offset + 2u]),
        glu_bf16_to_float(upScalar[offset + 3u])
    );
    float4 result = glu_float4(gateVec, upVec);

    for (uint lane = 0; lane < tail; ++lane) {
        destinationScalar[offset + lane] = glu_float_to_bf16(result[lane]);
    }
}

kernel void glu_bfloat16(
    device ushort4* destination [[buffer(0)]],
    device const ushort4* gateVector [[buffer(1)]],
    device const ushort4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    glu_bfloat16_kernel(destination, gateVector, upVector, count, index);
}

// --- activation_linglu.metal ---
#include <metal_stdlib>

using namespace metal;

// gate * up — matches pkg/backend/device/cpu/math.FastLinGLU32.
static inline float4 linglu_float4(float4 gate, float4 up) {
    return gate * up;
}

static inline float linglu_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort linglu_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

static inline void linglu_write_tail(
    device float* destination,
    uint offset,
    uint tail,
    float4 result
) {
    for (uint lane = 0; lane < tail; ++lane) {
        destination[offset + lane] = result[lane];
    }
}

static inline void linglu_float32_kernel(
    device float4* destination [[buffer(0)]],
    device const float4* gateVector [[buffer(1)]],
    device const float4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    device float* destinationScalar = (device float*)destination;
    device const float* gateScalar = (device const float*)gateVector;
    device const float* upScalar = (device const float*)upVector;
    uint offset = index * 4u;

    if (offset + 4u <= count) {
        destination[index] = linglu_float4(gateVector[index], upVector[index]);
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    float4 gateVec = float4(
        gateScalar[offset],
        gateScalar[offset + 1u],
        gateScalar[offset + 2u],
        gateScalar[offset + 3u]
    );
    float4 upVec = float4(
        upScalar[offset],
        upScalar[offset + 1u],
        upScalar[offset + 2u],
        upScalar[offset + 3u]
    );
    float4 result = linglu_float4(gateVec, upVec);

    linglu_write_tail(destinationScalar, offset, tail, result);
}

kernel void linglu_float32(
    device float4* destination [[buffer(0)]],
    device const float4* gateVector [[buffer(1)]],
    device const float4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    linglu_float32_kernel(destination, gateVector, upVector, count, index);
}

static inline void linglu_float16_kernel(
    device half4* destination [[buffer(0)]],
    device const half4* gateVector [[buffer(1)]],
    device const half4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    device half* destinationScalar = (device half*)destination;
    device const half* gateScalar = (device const half*)gateVector;
    device const half* upScalar = (device const half*)upVector;
    uint offset = index * 4u;

    if (offset + 4u <= count) {
        float4 gateVec = float4(gateVector[index]);
        float4 upVec = float4(upVector[index]);
        destination[index] = half4(linglu_float4(gateVec, upVec));
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    float4 gateVec = float4(
        float(gateScalar[offset]),
        float(gateScalar[offset + 1u]),
        float(gateScalar[offset + 2u]),
        float(gateScalar[offset + 3u])
    );
    float4 upVec = float4(
        float(upScalar[offset]),
        float(upScalar[offset + 1u]),
        float(upScalar[offset + 2u]),
        float(upScalar[offset + 3u])
    );
    float4 result = linglu_float4(gateVec, upVec);

    for (uint lane = 0; lane < tail; ++lane) {
        destinationScalar[offset + lane] = half(result[lane]);
    }
}

kernel void linglu_float16(
    device half4* destination [[buffer(0)]],
    device const half4* gateVector [[buffer(1)]],
    device const half4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    linglu_float16_kernel(destination, gateVector, upVector, count, index);
}

static inline void linglu_bfloat16_kernel(
    device ushort4* destination [[buffer(0)]],
    device const ushort4* gateVector [[buffer(1)]],
    device const ushort4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    device ushort* destinationScalar = (device ushort*)destination;
    device const ushort* gateScalar = (device const ushort*)gateVector;
    device const ushort* upScalar = (device const ushort*)upVector;
    uint offset = index * 4u;

    if (offset + 4u <= count) {
        float4 gateVec = float4(
            linglu_bf16_to_float(gateVector[index].x),
            linglu_bf16_to_float(gateVector[index].y),
            linglu_bf16_to_float(gateVector[index].z),
            linglu_bf16_to_float(gateVector[index].w)
        );
        float4 upVec = float4(
            linglu_bf16_to_float(upVector[index].x),
            linglu_bf16_to_float(upVector[index].y),
            linglu_bf16_to_float(upVector[index].z),
            linglu_bf16_to_float(upVector[index].w)
        );
        float4 result = linglu_float4(gateVec, upVec);
        destination[index] = ushort4(
            linglu_float_to_bf16(result.x),
            linglu_float_to_bf16(result.y),
            linglu_float_to_bf16(result.z),
            linglu_float_to_bf16(result.w)
        );
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    float4 gateVec = float4(
        linglu_bf16_to_float(gateScalar[offset]),
        linglu_bf16_to_float(gateScalar[offset + 1u]),
        linglu_bf16_to_float(gateScalar[offset + 2u]),
        linglu_bf16_to_float(gateScalar[offset + 3u])
    );
    float4 upVec = float4(
        linglu_bf16_to_float(upScalar[offset]),
        linglu_bf16_to_float(upScalar[offset + 1u]),
        linglu_bf16_to_float(upScalar[offset + 2u]),
        linglu_bf16_to_float(upScalar[offset + 3u])
    );
    float4 result = linglu_float4(gateVec, upVec);

    for (uint lane = 0; lane < tail; ++lane) {
        destinationScalar[offset + lane] = linglu_float_to_bf16(result[lane]);
    }
}

kernel void linglu_bfloat16(
    device ushort4* destination [[buffer(0)]],
    device const ushort4* gateVector [[buffer(1)]],
    device const ushort4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    linglu_bfloat16_kernel(destination, gateVector, upVector, count, index);
}

// --- activation_reglu.metal ---
#include <metal_stdlib>

using namespace metal;

// Matches pkg/backend/device/cpu/math.FastReLU32 (vector max for ReGLU).
static inline float4 metal_fast_relu32_float4(float4 value) {
    return max(value, 0.0f);
}

// gate * FastReLU32(up) — matches pkg/backend/device/cpu/math.FastReGLU32.
static inline float4 reglu_float4(float4 gate, float4 up) {
    return gate * metal_fast_relu32_float4(up);
}

static inline float reglu_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort reglu_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

static inline void reglu_write_tail(
    device float* destination,
    uint offset,
    uint tail,
    float4 result
) {
    for (uint lane = 0; lane < tail; ++lane) {
        destination[offset + lane] = result[lane];
    }
}

static inline void reglu_float32_kernel(
    device float4* destination [[buffer(0)]],
    device const float4* gateVector [[buffer(1)]],
    device const float4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    device float* destinationScalar = (device float*)destination;
    device const float* gateScalar = (device const float*)gateVector;
    device const float* upScalar = (device const float*)upVector;
    uint offset = index * 4u;

    if (offset + 4u <= count) {
        destination[index] = reglu_float4(gateVector[index], upVector[index]);
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    float4 gateVec = float4(
        gateScalar[offset],
        gateScalar[offset + 1u],
        gateScalar[offset + 2u],
        gateScalar[offset + 3u]
    );
    float4 upVec = float4(
        upScalar[offset],
        upScalar[offset + 1u],
        upScalar[offset + 2u],
        upScalar[offset + 3u]
    );
    float4 result = reglu_float4(gateVec, upVec);

    reglu_write_tail(destinationScalar, offset, tail, result);
}

kernel void reglu_float32(
    device float4* destination [[buffer(0)]],
    device const float4* gateVector [[buffer(1)]],
    device const float4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    reglu_float32_kernel(destination, gateVector, upVector, count, index);
}

static inline void reglu_float16_kernel(
    device half4* destination [[buffer(0)]],
    device const half4* gateVector [[buffer(1)]],
    device const half4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    device half* destinationScalar = (device half*)destination;
    device const half* gateScalar = (device const half*)gateVector;
    device const half* upScalar = (device const half*)upVector;
    uint offset = index * 4u;

    if (offset + 4u <= count) {
        float4 gateVec = float4(gateVector[index]);
        float4 upVec = float4(upVector[index]);
        destination[index] = half4(reglu_float4(gateVec, upVec));
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    float4 gateVec = float4(
        float(gateScalar[offset]),
        float(gateScalar[offset + 1u]),
        float(gateScalar[offset + 2u]),
        float(gateScalar[offset + 3u])
    );
    float4 upVec = float4(
        float(upScalar[offset]),
        float(upScalar[offset + 1u]),
        float(upScalar[offset + 2u]),
        float(upScalar[offset + 3u])
    );
    float4 result = reglu_float4(gateVec, upVec);

    for (uint lane = 0; lane < tail; ++lane) {
        destinationScalar[offset + lane] = half(result[lane]);
    }
}

kernel void reglu_float16(
    device half4* destination [[buffer(0)]],
    device const half4* gateVector [[buffer(1)]],
    device const half4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    reglu_float16_kernel(destination, gateVector, upVector, count, index);
}

static inline void reglu_bfloat16_kernel(
    device ushort4* destination [[buffer(0)]],
    device const ushort4* gateVector [[buffer(1)]],
    device const ushort4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    device ushort* destinationScalar = (device ushort*)destination;
    device const ushort* gateScalar = (device const ushort*)gateVector;
    device const ushort* upScalar = (device const ushort*)upVector;
    uint offset = index * 4u;

    if (offset + 4u <= count) {
        float4 gateVec = float4(
            reglu_bf16_to_float(gateVector[index].x),
            reglu_bf16_to_float(gateVector[index].y),
            reglu_bf16_to_float(gateVector[index].z),
            reglu_bf16_to_float(gateVector[index].w)
        );
        float4 upVec = float4(
            reglu_bf16_to_float(upVector[index].x),
            reglu_bf16_to_float(upVector[index].y),
            reglu_bf16_to_float(upVector[index].z),
            reglu_bf16_to_float(upVector[index].w)
        );
        float4 result = reglu_float4(gateVec, upVec);
        destination[index] = ushort4(
            reglu_float_to_bf16(result.x),
            reglu_float_to_bf16(result.y),
            reglu_float_to_bf16(result.z),
            reglu_float_to_bf16(result.w)
        );
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    float4 gateVec = float4(
        reglu_bf16_to_float(gateScalar[offset]),
        reglu_bf16_to_float(gateScalar[offset + 1u]),
        reglu_bf16_to_float(gateScalar[offset + 2u]),
        reglu_bf16_to_float(gateScalar[offset + 3u])
    );
    float4 upVec = float4(
        reglu_bf16_to_float(upScalar[offset]),
        reglu_bf16_to_float(upScalar[offset + 1u]),
        reglu_bf16_to_float(upScalar[offset + 2u]),
        reglu_bf16_to_float(upScalar[offset + 3u])
    );
    float4 result = reglu_float4(gateVec, upVec);

    for (uint lane = 0; lane < tail; ++lane) {
        destinationScalar[offset + lane] = reglu_float_to_bf16(result[lane]);
    }
}

kernel void reglu_bfloat16(
    device ushort4* destination [[buffer(0)]],
    device const ushort4* gateVector [[buffer(1)]],
    device const ushort4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    reglu_bfloat16_kernel(destination, gateVector, upVector, count, index);
}

// --- activation_seglu.metal ---
#include <metal_stdlib>

using namespace metal;

// Matches pkg/backend/device/cpu/math.FastExp32 (platform scalar for SeGLU).
static inline float metal_fast_exp32(float value) {
    if (value < -87.33654f) {
        return 0.0f;
    }

    if (value > 88.72283f) {
        return 0x1.fffffep127f;
    }

    float z = value * 1.4426950408889634f;
    int exponentK = int(z);

    if (z < 0.0f) {
        exponentK--;
    }

    float fraction = z - float(exponentK);
    float poly = 1.0f + fraction * (
        0.69314718f + fraction * (
            0.24022650f + fraction * (
                0.05550410f + fraction * (
                    0.00961812f + fraction * 0.00133389f
                )
            )
        )
    );

    uint bits = as_type<uint>(poly) + uint(exponentK) * 8388608u;
    return as_type<float>(bits);
}

// Matches pkg/backend/device/cpu/math.FastSigmoid32 (platform scalar for SeGLU).
static inline float metal_fast_sigmoid32(float value) {
    if (value >= 0.0f) {
        return 1.0f / (1.0f + metal_fast_exp32(-value));
    }

    float expValue = metal_fast_exp32(value);
    return expValue / (1.0f + expValue);
}

static inline float4 metal_fast_sigmoid32_float4(float4 value) {
    return float4(
        metal_fast_sigmoid32(value.x),
        metal_fast_sigmoid32(value.y),
        metal_fast_sigmoid32(value.z),
        metal_fast_sigmoid32(value.w)
    );
}

// up * FastSigmoid32(gate) — matches pkg/backend/device/cpu/math.FastSeGLU32.
static inline float4 seglu_float4(float4 gate, float4 up) {
    return up * metal_fast_sigmoid32_float4(gate);
}

static inline float seglu_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort seglu_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

static inline void seglu_write_tail(
    device float* destination,
    uint offset,
    uint tail,
    float4 result
) {
    for (uint lane = 0; lane < tail; ++lane) {
        destination[offset + lane] = result[lane];
    }
}

static inline void seglu_float32_kernel(
    device float4* destination [[buffer(0)]],
    device const float4* gateVector [[buffer(1)]],
    device const float4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    device float* destinationScalar = (device float*)destination;
    device const float* gateScalar = (device const float*)gateVector;
    device const float* upScalar = (device const float*)upVector;
    uint offset = index * 4u;

    if (offset + 4u <= count) {
        destination[index] = seglu_float4(gateVector[index], upVector[index]);
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    float4 gateVec = float4(
        gateScalar[offset],
        gateScalar[offset + 1u],
        gateScalar[offset + 2u],
        gateScalar[offset + 3u]
    );
    float4 upVec = float4(
        upScalar[offset],
        upScalar[offset + 1u],
        upScalar[offset + 2u],
        upScalar[offset + 3u]
    );
    float4 result = seglu_float4(gateVec, upVec);

    seglu_write_tail(destinationScalar, offset, tail, result);
}

kernel void seglu_float32(
    device float4* destination [[buffer(0)]],
    device const float4* gateVector [[buffer(1)]],
    device const float4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    seglu_float32_kernel(destination, gateVector, upVector, count, index);
}

static inline void seglu_float16_kernel(
    device half4* destination [[buffer(0)]],
    device const half4* gateVector [[buffer(1)]],
    device const half4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    device half* destinationScalar = (device half*)destination;
    device const half* gateScalar = (device const half*)gateVector;
    device const half* upScalar = (device const half*)upVector;
    uint offset = index * 4u;

    if (offset + 4u <= count) {
        float4 gateVec = float4(gateVector[index]);
        float4 upVec = float4(upVector[index]);
        destination[index] = half4(seglu_float4(gateVec, upVec));
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    float4 gateVec = float4(
        float(gateScalar[offset]),
        float(gateScalar[offset + 1u]),
        float(gateScalar[offset + 2u]),
        float(gateScalar[offset + 3u])
    );
    float4 upVec = float4(
        float(upScalar[offset]),
        float(upScalar[offset + 1u]),
        float(upScalar[offset + 2u]),
        float(upScalar[offset + 3u])
    );
    float4 result = seglu_float4(gateVec, upVec);

    for (uint lane = 0; lane < tail; ++lane) {
        destinationScalar[offset + lane] = half(result[lane]);
    }
}

kernel void seglu_float16(
    device half4* destination [[buffer(0)]],
    device const half4* gateVector [[buffer(1)]],
    device const half4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    seglu_float16_kernel(destination, gateVector, upVector, count, index);
}

static inline void seglu_bfloat16_kernel(
    device ushort4* destination [[buffer(0)]],
    device const ushort4* gateVector [[buffer(1)]],
    device const ushort4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    device ushort* destinationScalar = (device ushort*)destination;
    device const ushort* gateScalar = (device const ushort*)gateVector;
    device const ushort* upScalar = (device const ushort*)upVector;
    uint offset = index * 4u;

    if (offset + 4u <= count) {
        float4 gateVec = float4(
            seglu_bf16_to_float(gateVector[index].x),
            seglu_bf16_to_float(gateVector[index].y),
            seglu_bf16_to_float(gateVector[index].z),
            seglu_bf16_to_float(gateVector[index].w)
        );
        float4 upVec = float4(
            seglu_bf16_to_float(upVector[index].x),
            seglu_bf16_to_float(upVector[index].y),
            seglu_bf16_to_float(upVector[index].z),
            seglu_bf16_to_float(upVector[index].w)
        );
        float4 result = seglu_float4(gateVec, upVec);
        destination[index] = ushort4(
            seglu_float_to_bf16(result.x),
            seglu_float_to_bf16(result.y),
            seglu_float_to_bf16(result.z),
            seglu_float_to_bf16(result.w)
        );
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    float4 gateVec = float4(
        seglu_bf16_to_float(gateScalar[offset]),
        seglu_bf16_to_float(gateScalar[offset + 1u]),
        seglu_bf16_to_float(gateScalar[offset + 2u]),
        seglu_bf16_to_float(gateScalar[offset + 3u])
    );
    float4 upVec = float4(
        seglu_bf16_to_float(upScalar[offset]),
        seglu_bf16_to_float(upScalar[offset + 1u]),
        seglu_bf16_to_float(upScalar[offset + 2u]),
        seglu_bf16_to_float(upScalar[offset + 3u])
    );
    float4 result = seglu_float4(gateVec, upVec);

    for (uint lane = 0; lane < tail; ++lane) {
        destinationScalar[offset + lane] = seglu_float_to_bf16(result[lane]);
    }
}

kernel void seglu_bfloat16(
    device ushort4* destination [[buffer(0)]],
    device const ushort4* gateVector [[buffer(1)]],
    device const ushort4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    seglu_bfloat16_kernel(destination, gateVector, upVector, count, index);
}

// --- activation_siglu.metal ---
#include <metal_stdlib>

using namespace metal;

// Matches pkg/backend/device/cpu/math.FastExp32 (platform scalar for SiGLU).
static inline float metal_fast_exp32(float value) {
    if (value < -87.33654f) {
        return 0.0f;
    }

    if (value > 88.72283f) {
        return 0x1.fffffep127f;
    }

    float z = value * 1.4426950408889634f;
    int exponentK = int(z);

    if (z < 0.0f) {
        exponentK--;
    }

    float fraction = z - float(exponentK);
    float poly = 1.0f + fraction * (
        0.69314718f + fraction * (
            0.24022650f + fraction * (
                0.05550410f + fraction * (
                    0.00961812f + fraction * 0.00133389f
                )
            )
        )
    );

    uint bits = as_type<uint>(poly) + uint(exponentK) * 8388608u;
    return as_type<float>(bits);
}

// Matches pkg/backend/device/cpu/math.FastSigmoid32 (platform scalar for SiGLU).
static inline float metal_fast_sigmoid32(float value) {
    if (value >= 0.0f) {
        return 1.0f / (1.0f + metal_fast_exp32(-value));
    }

    float expValue = metal_fast_exp32(value);
    return expValue / (1.0f + expValue);
}

static inline float4 metal_fast_sigmoid32_float4(float4 value) {
    return float4(
        metal_fast_sigmoid32(value.x),
        metal_fast_sigmoid32(value.y),
        metal_fast_sigmoid32(value.z),
        metal_fast_sigmoid32(value.w)
    );
}

// FastSigmoid32(gate) * up — matches pkg/backend/device/cpu/math.FastSiGLU32.
static inline float4 siglu_float4(float4 gate, float4 up) {
    return metal_fast_sigmoid32_float4(gate) * up;
}

static inline float siglu_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort siglu_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

static inline void siglu_write_tail(
    device float* destination,
    uint offset,
    uint tail,
    float4 result
) {
    for (uint lane = 0; lane < tail; ++lane) {
        destination[offset + lane] = result[lane];
    }
}

static inline void siglu_float32_kernel(
    device float4* destination [[buffer(0)]],
    device const float4* gateVector [[buffer(1)]],
    device const float4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    device float* destinationScalar = (device float*)destination;
    device const float* gateScalar = (device const float*)gateVector;
    device const float* upScalar = (device const float*)upVector;
    uint offset = index * 4u;

    if (offset + 4u <= count) {
        destination[index] = siglu_float4(gateVector[index], upVector[index]);
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    float4 gateVec = float4(
        gateScalar[offset],
        gateScalar[offset + 1u],
        gateScalar[offset + 2u],
        gateScalar[offset + 3u]
    );
    float4 upVec = float4(
        upScalar[offset],
        upScalar[offset + 1u],
        upScalar[offset + 2u],
        upScalar[offset + 3u]
    );
    float4 result = siglu_float4(gateVec, upVec);

    siglu_write_tail(destinationScalar, offset, tail, result);
}

kernel void siglu_float32(
    device float4* destination [[buffer(0)]],
    device const float4* gateVector [[buffer(1)]],
    device const float4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    siglu_float32_kernel(destination, gateVector, upVector, count, index);
}

static inline void siglu_float16_kernel(
    device half4* destination [[buffer(0)]],
    device const half4* gateVector [[buffer(1)]],
    device const half4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    device half* destinationScalar = (device half*)destination;
    device const half* gateScalar = (device const half*)gateVector;
    device const half* upScalar = (device const half*)upVector;
    uint offset = index * 4u;

    if (offset + 4u <= count) {
        float4 gateVec = float4(gateVector[index]);
        float4 upVec = float4(upVector[index]);
        destination[index] = half4(siglu_float4(gateVec, upVec));
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    float4 gateVec = float4(
        float(gateScalar[offset]),
        float(gateScalar[offset + 1u]),
        float(gateScalar[offset + 2u]),
        float(gateScalar[offset + 3u])
    );
    float4 upVec = float4(
        float(upScalar[offset]),
        float(upScalar[offset + 1u]),
        float(upScalar[offset + 2u]),
        float(upScalar[offset + 3u])
    );
    float4 result = siglu_float4(gateVec, upVec);

    for (uint lane = 0; lane < tail; ++lane) {
        destinationScalar[offset + lane] = half(result[lane]);
    }
}

kernel void siglu_float16(
    device half4* destination [[buffer(0)]],
    device const half4* gateVector [[buffer(1)]],
    device const half4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    siglu_float16_kernel(destination, gateVector, upVector, count, index);
}

static inline void siglu_bfloat16_kernel(
    device ushort4* destination [[buffer(0)]],
    device const ushort4* gateVector [[buffer(1)]],
    device const ushort4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    device ushort* destinationScalar = (device ushort*)destination;
    device const ushort* gateScalar = (device const ushort*)gateVector;
    device const ushort* upScalar = (device const ushort*)upVector;
    uint offset = index * 4u;

    if (offset + 4u <= count) {
        float4 gateVec = float4(
            siglu_bf16_to_float(gateVector[index].x),
            siglu_bf16_to_float(gateVector[index].y),
            siglu_bf16_to_float(gateVector[index].z),
            siglu_bf16_to_float(gateVector[index].w)
        );
        float4 upVec = float4(
            siglu_bf16_to_float(upVector[index].x),
            siglu_bf16_to_float(upVector[index].y),
            siglu_bf16_to_float(upVector[index].z),
            siglu_bf16_to_float(upVector[index].w)
        );
        float4 result = siglu_float4(gateVec, upVec);
        destination[index] = ushort4(
            siglu_float_to_bf16(result.x),
            siglu_float_to_bf16(result.y),
            siglu_float_to_bf16(result.z),
            siglu_float_to_bf16(result.w)
        );
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    float4 gateVec = float4(
        siglu_bf16_to_float(gateScalar[offset]),
        siglu_bf16_to_float(gateScalar[offset + 1u]),
        siglu_bf16_to_float(gateScalar[offset + 2u]),
        siglu_bf16_to_float(gateScalar[offset + 3u])
    );
    float4 upVec = float4(
        siglu_bf16_to_float(upScalar[offset]),
        siglu_bf16_to_float(upScalar[offset + 1u]),
        siglu_bf16_to_float(upScalar[offset + 2u]),
        siglu_bf16_to_float(upScalar[offset + 3u])
    );
    float4 result = siglu_float4(gateVec, upVec);

    for (uint lane = 0; lane < tail; ++lane) {
        destinationScalar[offset + lane] = siglu_float_to_bf16(result[lane]);
    }
}

kernel void siglu_bfloat16(
    device ushort4* destination [[buffer(0)]],
    device const ushort4* gateVector [[buffer(1)]],
    device const ushort4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    siglu_bfloat16_kernel(destination, gateVector, upVector, count, index);
}

// --- activation_swiglu.metal ---
#include <metal_stdlib>

using namespace metal;

// Matches pkg/backend/device/cpu/math.FastExp32 (platform scalar for SwiGLU).
static inline float metal_fast_exp32(float value) {
    if (value < -87.33654f) {
        return 0.0f;
    }

    if (value > 88.72283f) {
        return 0x1.fffffep127f;
    }

    float z = value * 1.4426950408889634f;
    int32_t exponentK = int32_t(z);

    if (z < 0.0f) {
        exponentK--;
    }

    float fraction = z - float(exponentK);
    float poly = 1.0f + fraction * (
        0.69314718f + fraction * (
            0.24022650f + fraction * (
                0.05550410f + fraction * (
                    0.00961812f + fraction * 0.00133389f
                )
            )
        )
    );

    uint bits = as_type<uint>(poly) + (as_type<uint>(exponentK) << 23);
    return as_type<float>(bits);
}

static inline float4 metal_fast_exp32_float4(float4 value) {
    return float4(
        metal_fast_exp32(value.x),
        metal_fast_exp32(value.y),
        metal_fast_exp32(value.z),
        metal_fast_exp32(value.w)
    );
}

static inline float4 swiglu_float4(float4 gate, float4 up) {
    float4 silu = gate / (float4(1.0f) + metal_fast_exp32_float4(-gate));
    return fma(silu, up, float4(0.0f));
}

static inline float swiglu_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort swiglu_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

static inline uint swiglu_packed_source_index(uint outputIndex, uint inner, uint sideOffset) {
    uint row = outputIndex / inner;
    uint column = outputIndex - row * inner;

    return row * inner * 2u + sideOffset + column;
}

static inline void swiglu_write_tail(
    device float* destination,
    uint offset,
    uint tail,
    float4 result
) {
    for (uint lane = 0; lane < tail; ++lane) {
        destination[offset + lane] = result[lane];
    }
}

static inline void swiglu_float32_kernel(
    device float4* destination [[buffer(0)]],
    device const float4* gateVector [[buffer(1)]],
    device const float4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    device float* destinationScalar = (device float*)destination;
    device const float* gateScalar = (device const float*)gateVector;
    device const float* upScalar = (device const float*)upVector;
    uint offset = index * 4u;

    if (offset + 4u <= count) {
        destination[index] = swiglu_float4(gateVector[index], upVector[index]);
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    float4 gateVec = float4(
        gateScalar[offset],
        gateScalar[offset + 1u],
        gateScalar[offset + 2u],
        gateScalar[offset + 3u]
    );
    float4 upVec = float4(
        upScalar[offset],
        upScalar[offset + 1u],
        upScalar[offset + 2u],
        upScalar[offset + 3u]
    );
    float4 result = swiglu_float4(gateVec, upVec);

    swiglu_write_tail(destinationScalar, offset, tail, result);
}

kernel void swiglu_float32(
    device float4* destination [[buffer(0)]],
    device const float4* gateVector [[buffer(1)]],
    device const float4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    swiglu_float32_kernel(destination, gateVector, upVector, count, index);
}

kernel void swiglu_packed_float32(
    device float4* destination [[buffer(0)]],
    device const float* packed [[buffer(1)]],
    constant uint& inner [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    device float* destinationScalar = (device float*)destination;
    uint offset = index * 4u;

    if (offset >= count) {
        return;
    }

    float4 gateVec;
    float4 upVec;

    for (uint lane = 0; lane < 4u; lane++) {
        uint outputIndex = offset + lane;

        if (outputIndex >= count) {
            break;
        }

        uint gateIndex = swiglu_packed_source_index(outputIndex, inner, 0u);
        gateVec[lane] = packed[gateIndex];
        upVec[lane] = packed[gateIndex + inner];
    }

    float4 result = swiglu_float4(gateVec, upVec);
    uint tail = min(4u, count - offset);

    swiglu_write_tail(destinationScalar, offset, tail, result);
}

static inline void swiglu_float16_kernel(
    device half4* destination [[buffer(0)]],
    device const half4* gateVector [[buffer(1)]],
    device const half4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    device half* destinationScalar = (device half*)destination;
    device const half* gateScalar = (device const half*)gateVector;
    device const half* upScalar = (device const half*)upVector;
    uint offset = index * 4u;

    if (offset + 4u <= count) {
        float4 gateVec = float4(gateVector[index]);
        float4 upVec = float4(upVector[index]);
        destination[index] = half4(swiglu_float4(gateVec, upVec));
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    float4 gateVec = float4(
        float(gateScalar[offset]),
        float(gateScalar[offset + 1u]),
        float(gateScalar[offset + 2u]),
        float(gateScalar[offset + 3u])
    );
    float4 upVec = float4(
        float(upScalar[offset]),
        float(upScalar[offset + 1u]),
        float(upScalar[offset + 2u]),
        float(upScalar[offset + 3u])
    );
    float4 result = swiglu_float4(gateVec, upVec);

    for (uint lane = 0; lane < tail; ++lane) {
        destinationScalar[offset + lane] = half(result[lane]);
    }
}

kernel void swiglu_float16(
    device half4* destination [[buffer(0)]],
    device const half4* gateVector [[buffer(1)]],
    device const half4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    swiglu_float16_kernel(destination, gateVector, upVector, count, index);
}

kernel void swiglu_packed_float16(
    device half4* destination [[buffer(0)]],
    device const half* packed [[buffer(1)]],
    constant uint& inner [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    device half* destinationScalar = (device half*)destination;
    uint offset = index * 4u;

    if (offset >= count) {
        return;
    }

    float4 gateVec;
    float4 upVec;

    for (uint lane = 0; lane < 4u; lane++) {
        uint outputIndex = offset + lane;

        if (outputIndex >= count) {
            break;
        }

        uint gateIndex = swiglu_packed_source_index(outputIndex, inner, 0u);
        gateVec[lane] = float(packed[gateIndex]);
        upVec[lane] = float(packed[gateIndex + inner]);
    }

    float4 result = swiglu_float4(gateVec, upVec);
    uint tail = min(4u, count - offset);

    for (uint lane = 0; lane < tail; ++lane) {
        destinationScalar[offset + lane] = half(result[lane]);
    }
}

static inline void swiglu_bfloat16_kernel(
    device ushort4* destination [[buffer(0)]],
    device const ushort4* gateVector [[buffer(1)]],
    device const ushort4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    device ushort* destinationScalar = (device ushort*)destination;
    device const ushort* gateScalar = (device const ushort*)gateVector;
    device const ushort* upScalar = (device const ushort*)upVector;
    uint offset = index * 4u;

    if (offset + 4u <= count) {
        float4 gateVec = float4(
            swiglu_bf16_to_float(gateVector[index].x),
            swiglu_bf16_to_float(gateVector[index].y),
            swiglu_bf16_to_float(gateVector[index].z),
            swiglu_bf16_to_float(gateVector[index].w)
        );
        float4 upVec = float4(
            swiglu_bf16_to_float(upVector[index].x),
            swiglu_bf16_to_float(upVector[index].y),
            swiglu_bf16_to_float(upVector[index].z),
            swiglu_bf16_to_float(upVector[index].w)
        );
        float4 result = swiglu_float4(gateVec, upVec);
        destination[index] = ushort4(
            swiglu_float_to_bf16(result.x),
            swiglu_float_to_bf16(result.y),
            swiglu_float_to_bf16(result.z),
            swiglu_float_to_bf16(result.w)
        );
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    float4 gateVec = float4(
        swiglu_bf16_to_float(gateScalar[offset]),
        swiglu_bf16_to_float(gateScalar[offset + 1u]),
        swiglu_bf16_to_float(gateScalar[offset + 2u]),
        swiglu_bf16_to_float(gateScalar[offset + 3u])
    );
    float4 upVec = float4(
        swiglu_bf16_to_float(upScalar[offset]),
        swiglu_bf16_to_float(upScalar[offset + 1u]),
        swiglu_bf16_to_float(upScalar[offset + 2u]),
        swiglu_bf16_to_float(upScalar[offset + 3u])
    );
    float4 result = swiglu_float4(gateVec, upVec);

    for (uint lane = 0; lane < tail; ++lane) {
        destinationScalar[offset + lane] = swiglu_float_to_bf16(result[lane]);
    }
}

kernel void swiglu_bfloat16(
    device ushort4* destination [[buffer(0)]],
    device const ushort4* gateVector [[buffer(1)]],
    device const ushort4* upVector [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    swiglu_bfloat16_kernel(destination, gateVector, upVector, count, index);
}

kernel void swiglu_packed_bfloat16(
    device ushort4* destination [[buffer(0)]],
    device const ushort* packed [[buffer(1)]],
    constant uint& inner [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    device ushort* destinationScalar = (device ushort*)destination;
    uint offset = index * 4u;

    if (offset >= count) {
        return;
    }

    float4 gateVec;
    float4 upVec;

    for (uint lane = 0; lane < 4u; lane++) {
        uint outputIndex = offset + lane;

        if (outputIndex >= count) {
            break;
        }

        uint gateIndex = swiglu_packed_source_index(outputIndex, inner, 0u);
        gateVec[lane] = swiglu_bf16_to_float(packed[gateIndex]);
        upVec[lane] = swiglu_bf16_to_float(packed[gateIndex + inner]);
    }

    float4 result = swiglu_float4(gateVec, upVec);
    uint tail = min(4u, count - offset);

    for (uint lane = 0; lane < tail; ++lane) {
        destinationScalar[offset + lane] = swiglu_float_to_bf16(result[lane]);
    }
}

kernel void swiglu_silu_float32(
    device const float* inputVector [[buffer(0)]],
    device float* outVector [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) {
        return;
    }

    float value = inputVector[index];
    outVector[index] = value / (1.0f + metal_fast_exp32(-value));
}

