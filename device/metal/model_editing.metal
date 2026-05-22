#include <metal_stdlib>
using namespace metal;

static inline void weight_graft_write_tail(
    device float* weights,
    uint offset,
    uint tail,
    float4 result
) {
    for (uint lane = 0; lane < tail; ++lane) {
        weights[offset + lane] = result[lane];
    }
}

kernel void weight_graft_add_float32(
    device float4* weights [[buffer(0)]],
    device const float4* injection [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    device float* weightsScalar = (device float*)weights;
    device const float* injectionScalar = (device const float*)injection;
    uint offset = index * 4u;

    if (offset + 4u <= count) {
        weights[index] = weights[index] + injection[index];
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    float4 weightVec = float4(
        weightsScalar[offset],
        weightsScalar[offset + 1u],
        weightsScalar[offset + 2u],
        weightsScalar[offset + 3u]
    );
    float4 injectionVec = float4(
        injectionScalar[offset],
        injectionScalar[offset + 1u],
        injectionScalar[offset + 2u],
        injectionScalar[offset + 3u]
    );
    float4 result = weightVec + injectionVec;

    weight_graft_write_tail(weightsScalar, offset, tail, result);
}

kernel void weight_graft_add_float16(
    device half4* weights [[buffer(0)]],
    device const half4* injection [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    device half* weightsScalar = (device half*)weights;
    device const half* injectionScalar = (device const half*)injection;
    uint offset = index * 4u;

    if (offset + 4u <= count) {
        weights[index] = weights[index] + injection[index];
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    half4 weightVec = half4(
        weightsScalar[offset],
        weightsScalar[offset + 1u],
        weightsScalar[offset + 2u],
        weightsScalar[offset + 3u]
    );
    half4 injectionVec = half4(
        injectionScalar[offset],
        injectionScalar[offset + 1u],
        injectionScalar[offset + 2u],
        injectionScalar[offset + 3u]
    );
    half4 result = weightVec + injectionVec;

    for (uint lane = 0; lane < tail; ++lane) {
        weightsScalar[offset + lane] = result[lane];
    }
}

kernel void weight_graft_add_bfloat16(
    device ushort4* weights [[buffer(0)]],
    device const ushort4* injection [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    device ushort* weightsScalar = (device ushort*)weights;
    device const ushort* injectionScalar = (device const ushort*)injection;
    uint offset = index * 4u;

    if (offset + 4u <= count) {
        float4 weightFloat = float4(
            as_type<float>(uint(weights[index].x) << 16),
            as_type<float>(uint(weights[index].y) << 16),
            as_type<float>(uint(weights[index].z) << 16),
            as_type<float>(uint(weights[index].w) << 16)
        );
        float4 injectionFloat = float4(
            as_type<float>(uint(injection[index].x) << 16),
            as_type<float>(uint(injection[index].y) << 16),
            as_type<float>(uint(injection[index].z) << 16),
            as_type<float>(uint(injection[index].w) << 16)
        );
        float4 sum = weightFloat + injectionFloat;
        weights[index] = ushort4(
            ushort(as_type<uint>(sum.x) >> 16),
            ushort(as_type<uint>(sum.y) >> 16),
            ushort(as_type<uint>(sum.z) >> 16),
            ushort(as_type<uint>(sum.w) >> 16)
        );
        return;
    }

    if (offset >= count) {
        return;
    }

    uint tail = count - offset;
    float4 weightFloat = float4(0.0f);
    float4 injectionFloat = float4(0.0f);

    for (uint lane = 0; lane < tail; ++lane) {
        weightFloat[lane] = as_type<float>(uint(weightsScalar[offset + lane]) << 16);
        injectionFloat[lane] = as_type<float>(uint(injectionScalar[offset + lane]) << 16);
    }

    float4 sum = weightFloat + injectionFloat;

    for (uint lane = 0; lane < tail; ++lane) {
        weightsScalar[offset + lane] = ushort(as_type<uint>(sum[lane]) >> 16);
    }
}
