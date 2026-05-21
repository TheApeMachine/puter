#include <metal_stdlib>

using namespace metal;

static inline float axpy_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort axpy_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

kernel void axpy_float32(
    device float* y [[buffer(0)]],
    device const float* x [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant float& alpha [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint stride [[threads_per_grid]]
) {
    uint vectorCount = count / 4;
    device float4* yVector = reinterpret_cast<device float4*>(y);
    device const float4* xVector = reinterpret_cast<device const float4*>(x);

    for (uint i = index; i < vectorCount; i += stride) {
        yVector[i] += float(alpha) * xVector[i];
    }

    if (index == 0) {
        uint remainder = count % 4;
        for (uint offset = 0; offset < remainder; offset++) {
            uint scalarIndex = vectorCount * 4 + offset;
            y[scalarIndex] += float(alpha) * x[scalarIndex];
        }
    }
}

kernel void axpy_bfloat16(
    device ushort4* yVector [[buffer(0)]],
    device const ushort4* xVector [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant float& alpha [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint stride [[threads_per_grid]]
) {
    uint vectorCount = count / 4;

    for (uint vectorIndex = index; vectorIndex < vectorCount; vectorIndex += stride) {
        ushort4 yPacked = yVector[vectorIndex];
        ushort4 xPacked = xVector[vectorIndex];
        float4 yValues = float4(
            axpy_bf16_to_float(yPacked.x),
            axpy_bf16_to_float(yPacked.y),
            axpy_bf16_to_float(yPacked.z),
            axpy_bf16_to_float(yPacked.w)
        );
        float4 xValues = float4(
            axpy_bf16_to_float(xPacked.x),
            axpy_bf16_to_float(xPacked.y),
            axpy_bf16_to_float(xPacked.z),
            axpy_bf16_to_float(xPacked.w)
        );
        float4 result = yValues + float(alpha) * xValues;
        yVector[vectorIndex] = ushort4(
            axpy_float_to_bf16(result.x),
            axpy_float_to_bf16(result.y),
            axpy_float_to_bf16(result.z),
            axpy_float_to_bf16(result.w)
        );
    }

    if (index == 0) {
        uint remainder = count % 4;
        device ushort* yScalar = reinterpret_cast<device ushort*>(yVector);
        device const ushort* xScalar = reinterpret_cast<device const ushort*>(xVector);

        for (uint offset = 0; offset < remainder; offset++) {
            uint scalarIndex = vectorCount * 4 + offset;
            float result = axpy_bf16_to_float(yScalar[scalarIndex]) +
                float(alpha) * axpy_bf16_to_float(xScalar[scalarIndex]);
            yScalar[scalarIndex] = axpy_float_to_bf16(result);
        }
    }
}

kernel void dot_float32(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device atomic_float* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]],
    uint stride [[threads_per_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]]
) {
    float partial = 0.0f;

    for (uint offset = index; offset < count; offset += stride) {
        partial += left[offset] * right[offset];
    }

    float simd_partial = simd_sum(partial);

    if (simd_lane_id == 0) {
        atomic_fetch_add_explicit(out, simd_partial, memory_order_relaxed);
    }
}
