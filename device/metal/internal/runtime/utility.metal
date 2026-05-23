#include <metal_stdlib>

using namespace metal;

static inline void write_u64_le(device uchar* out, uint offset, ulong value) {
    for (uint byteIndex = 0; byteIndex < 8; byteIndex++) {
        out[offset + byteIndex] = uchar((value >> (byteIndex * 8)) & 0xfful);
    }
}

static inline void write_u32_le(device uchar* out, uint offset, uint value) {
    for (uint byteIndex = 0; byteIndex < 4; byteIndex++) {
        out[offset + byteIndex] = uchar((value >> (byteIndex * 8)) & 0xffu);
    }
}

static inline bool mask_bit(device const uchar* mask, uint index) {
    return ((mask[index >> 3u] >> (index & 7u)) & 1u) != 0u;
}

kernel void checkpoint_encode_float32(
    device const uint* inputBits [[buffer(0)]],
    device uchar* out [[buffer(1)]],
    constant uint& rank [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant ulong* dims [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    uint headerBytes = 16u + rank * 8u;

    if (index == 0u) {
        write_u64_le(out, 0u, ulong(rank));
        write_u64_le(out, 8u, ulong(count) * 4ul);

        for (uint dimIndex = 0u; dimIndex < rank; dimIndex++) {
            write_u64_le(out, 16u + dimIndex * 8u, dims[dimIndex]);
        }
    }

    if (index >= count) {
        return;
    }

    write_u32_le(out, headerBytes + index * 4u, inputBits[index]);
}

kernel void checkpoint_decode_float32(
    device const uchar* input [[buffer(0)]],
    device uint* outBits [[buffer(1)]],
    constant uint& headerBytes [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) {
        return;
    }

    device const uint* inputBits = reinterpret_cast<device const uint*>(input + headerBytes);
    outBits[index] = inputBits[index];
}

kernel void tokenizer_pack_int32(
    device const int4* input [[buffer(0)]],
    device int4* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    uint base = index * 4u;

    if (base + 3u < count) {
        out[index] = input[index];
        return;
    }

    device const int* inputScalar = reinterpret_cast<device const int*>(input);
    device int* outScalar = reinterpret_cast<device int*>(out);

    for (uint offset = 0u; offset < 4u; offset++) {
        uint elementIndex = base + offset;

        if (elementIndex < count) {
            outScalar[elementIndex] = inputScalar[elementIndex];
        }
    }
}

kernel void weight_freeze_mask_float32(
    device const uchar* mask [[buffer(0)]],
    device const float4* gradients [[buffer(1)]],
    device float4* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    uint base = index * 4u;
    float4 values = gradients[index];
    float4 result = float4(0.0f);

    if (base < count && mask_bit(mask, base)) {
        result.x = values.x;
    }

    if (base + 1u < count && mask_bit(mask, base + 1u)) {
        result.y = values.y;
    }

    if (base + 2u < count && mask_bit(mask, base + 2u)) {
        result.z = values.z;
    }

    if (base + 3u < count && mask_bit(mask, base + 3u)) {
        result.w = values.w;
    }

    out[index] = result;
}

kernel void weight_freeze_mask_float16(
    device const uchar* mask [[buffer(0)]],
    device const half4* gradients [[buffer(1)]],
    device half4* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    uint base = index * 4u;
    half4 values = gradients[index];
    half4 result = half4(0.0h);

    if (base < count && mask_bit(mask, base)) {
        result.x = values.x;
    }

    if (base + 1u < count && mask_bit(mask, base + 1u)) {
        result.y = values.y;
    }

    if (base + 2u < count && mask_bit(mask, base + 2u)) {
        result.z = values.z;
    }

    if (base + 3u < count && mask_bit(mask, base + 3u)) {
        result.w = values.w;
    }

    out[index] = result;
}

kernel void weight_freeze_mask_bfloat16(
    device const uchar* mask [[buffer(0)]],
    device const ushort4* gradients [[buffer(1)]],
    device ushort4* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    uint base = index * 4u;
    ushort4 values = gradients[index];
    ushort4 result = ushort4(0u);

    if (base < count && mask_bit(mask, base)) {
        result.x = values.x;
    }

    if (base + 1u < count && mask_bit(mask, base + 1u)) {
        result.y = values.y;
    }

    if (base + 2u < count && mask_bit(mask, base + 2u)) {
        result.z = values.z;
    }

    if (base + 3u < count && mask_bit(mask, base + 3u)) {
        result.w = values.w;
    }

    out[index] = result;
}

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

static inline float4 utility_bf16_to_float4(ushort4 value) {
    return float4(
        as_type<float>(uint(value.x) << 16),
        as_type<float>(uint(value.y) << 16),
        as_type<float>(uint(value.z) << 16),
        as_type<float>(uint(value.w) << 16)
    );
}

static inline ushort4 utility_float4_to_bf16(float4 value) {
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
        float4 baseFloat = utility_bf16_to_float4(baseVector[index]);
        float4 directionFloat = utility_bf16_to_float4(directionVector[index]);
        destination[index] = utility_float4_to_bf16(baseFloat + coefficient * directionFloat);
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
    float4 baseFloat = utility_bf16_to_float4(baseVec);
    float4 directionFloat = utility_bf16_to_float4(directionVec);
    ushort4 result = utility_float4_to_bf16(baseFloat + coefficient * directionFloat);

    activation_steer_write_tail_bf16(destinationScalar, offset, tail, result);
}
