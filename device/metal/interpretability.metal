#include <metal_stdlib>
using namespace metal;

static inline float4 activation_steer_float4(
    float4 baseVector,
    float4 directionVector,
    float coefficient
) {
    return baseVector + coefficient * directionVector;
}

static inline void activation_steer_write_tail(
    device float* destination,
    uint offset,
    uint tail,
    float4 result
) {
    for (uint lane = 0; lane < tail; ++lane) {
        destination[offset + lane] = result[lane];
    }
}

kernel void activation_steer_float32(
    device float4* destination [[buffer(0)]],
    device const float4* baseVector [[buffer(1)]],
    device const float4* directionVector [[buffer(2)]],
    constant float& coefficient [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    device float* destinationScalar = (device float*)destination;
    device const float* baseScalar = (device const float*)baseVector;
    device const float* directionScalar = (device const float*)directionVector;
    uint offset = index * 4u;

    if (offset + 4u <= count) {
        destination[index] = activation_steer_float4(
            baseVector[index],
            directionVector[index],
            coefficient
        );
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    float4 baseVec = float4(
        baseScalar[offset],
        baseScalar[offset + 1u],
        baseScalar[offset + 2u],
        baseScalar[offset + 3u]
    );
    float4 directionVec = float4(
        directionScalar[offset],
        directionScalar[offset + 1u],
        directionScalar[offset + 2u],
        directionScalar[offset + 3u]
    );
    float4 result = activation_steer_float4(baseVec, directionVec, coefficient);

    activation_steer_write_tail(destinationScalar, offset, tail, result);
}

static inline float4 activation_steer_half4(
    half4 baseVector,
    half4 directionVector,
    float coefficient
) {
    float4 baseFloat = float4(baseVector);
    float4 directionFloat = float4(directionVector);

    return baseFloat + coefficient * directionFloat;
}

static inline void activation_steer_write_tail_half(
    device half* destination,
    uint offset,
    uint tail,
    half4 result
) {
    for (uint lane = 0; lane < tail; ++lane) {
        destination[offset + lane] = result[lane];
    }
}

kernel void activation_steer_float16(
    device half4* destination [[buffer(0)]],
    device const half4* baseVector [[buffer(1)]],
    device const half4* directionVector [[buffer(2)]],
    constant float& coefficient [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    device half* destinationScalar = (device half*)destination;
    device const half* baseScalar = (device const half*)baseVector;
    device const half* directionScalar = (device const half*)directionVector;
    uint offset = index * 4u;

    if (offset + 4u <= count) {
        half4 baseVec = baseVector[index];
        half4 directionVec = directionVector[index];
        half4 result = half4(activation_steer_half4(baseVec, directionVec, coefficient));
        destination[index] = result;
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    half4 baseVec = half4(
        baseScalar[offset],
        baseScalar[offset + 1u],
        baseScalar[offset + 2u],
        baseScalar[offset + 3u]
    );
    half4 directionVec = half4(
        directionScalar[offset],
        directionScalar[offset + 1u],
        directionScalar[offset + 2u],
        directionScalar[offset + 3u]
    );
    half4 result = half4(activation_steer_half4(baseVec, directionVec, coefficient));

    activation_steer_write_tail_half(destinationScalar, offset, tail, result);
}

static inline float4 bf16_to_float4(ushort4 value) {
    return float4(
        as_type<float>(uint(value.x) << 16),
        as_type<float>(uint(value.y) << 16),
        as_type<float>(uint(value.z) << 16),
        as_type<float>(uint(value.w) << 16)
    );
}

static inline ushort4 float4_to_bf16(float4 value) {
    return ushort4(
        ushort(as_type<uint>(value.x) >> 16),
        ushort(as_type<uint>(value.y) >> 16),
        ushort(as_type<uint>(value.z) >> 16),
        ushort(as_type<uint>(value.w) >> 16)
    );
}

static inline void activation_steer_write_tail_bf16(
    device ushort* destination,
    uint offset,
    uint tail,
    ushort4 result
) {
    for (uint lane = 0; lane < tail; ++lane) {
        destination[offset + lane] = result[lane];
    }
}

kernel void activation_steer_bfloat16(
    device ushort4* destination [[buffer(0)]],
    device const ushort4* baseVector [[buffer(1)]],
    device const ushort4* directionVector [[buffer(2)]],
    constant float& coefficient [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    device ushort* destinationScalar = (device ushort*)destination;
    device const ushort* baseScalar = (device const ushort*)baseVector;
    device const ushort* directionScalar = (device const ushort*)directionVector;
    uint offset = index * 4u;

    if (offset + 4u <= count) {
        float4 baseFloat = bf16_to_float4(baseVector[index]);
        float4 directionFloat = bf16_to_float4(directionVector[index]);
        destination[index] = float4_to_bf16(baseFloat + coefficient * directionFloat);
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    ushort4 baseVec = ushort4(
        baseScalar[offset],
        baseScalar[offset + 1u],
        baseScalar[offset + 2u],
        baseScalar[offset + 3u]
    );
    ushort4 directionVec = ushort4(
        directionScalar[offset],
        directionScalar[offset + 1u],
        directionScalar[offset + 2u],
        directionScalar[offset + 3u]
    );
    float4 baseFloat = bf16_to_float4(baseVec);
    float4 directionFloat = bf16_to_float4(directionVec);
    ushort4 result = float4_to_bf16(baseFloat + coefficient * directionFloat);

    activation_steer_write_tail_bf16(destinationScalar, offset, tail, result);
}
