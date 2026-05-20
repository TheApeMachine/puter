#include <metal_stdlib>

using namespace metal;

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
