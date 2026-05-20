#include <metal_stdlib>

using namespace metal;

kernel void axpy_float32(
    device float* y [[buffer(0)]],
    device const float* x [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant float& alpha [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    uint base = index * 4;

    if (base + 3 < count) {
        device float4* yVector = reinterpret_cast<device float4*>(y);
        device const float4* xVector = reinterpret_cast<device const float4*>(x);
        yVector[index] += alpha * xVector[index];
        return;
    }

    for (uint offset = 0; offset < 4; offset++) {
        uint scalarIndex = base + offset;

        if (scalarIndex < count) {
            y[scalarIndex] += alpha * x[scalarIndex];
        }
    }
}

kernel void dot_float32(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    threadgroup float reduction[256];

    float partial = 0.0f;
    uint stride = 256;

    for (uint offset = index; offset < count; offset += stride) {
        partial += left[offset] * right[offset];
    }

    reduction[index] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint width = stride / 2; width > 0; width >>= 1) {
        if (index < width) {
            reduction[index] += reduction[index + width];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (index == 0) {
        out[0] = reduction[0];
    }
}
