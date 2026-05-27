#include "../layernorm/layernorm_common.metalinc"
#include "../internal/norm_stats.metalinc"

using namespace metal;

static inline void groupnorm_stats_rows_f32(
    device const float* input,
    device float* rowStats,
    threadgroup float* reduction,
    threadgroup ulong* sf64Reduction,
    constant uint& channels,
    constant uint& spatial,
    constant uint& groups,
    uint row,
    uint threadIndex
) {
    uint groupIndex = row % groups;
    uint batchIndex = row / groups;
    uint channelsPerGroup = channels / groups;
    uint channelStart = groupIndex * channelsPerGroup;
    uint groupSize = channelsPerGroup * spatial;
    uint groupOffset = (batchIndex * channels + channelStart) * spatial;
    ulong sum64 = SF64_NORM_ZERO;

    for (uint offset = threadIndex; offset < groupSize; offset += normalizationThreadCount) {
        float value = input[groupOffset + offset];
        sum64 = metal_sf64_add(sum64, metal_sf32_to64(as_type<uint>(value)));
    }

    sf64Reduction[threadIndex] = sum64;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = normalizationThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            sf64Reduction[threadIndex] = metal_sf64_add(
                sf64Reduction[threadIndex],
                sf64Reduction[threadIndex + stride]
            );
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean = metal_norm_mean_f32(sf64Reduction[0], groupSize);
    ulong variance64 = SF64_NORM_ZERO;

    for (uint offset = threadIndex; offset < groupSize; offset += normalizationThreadCount) {
        float delta = input[groupOffset + offset] - mean;
        ulong delta64 = metal_sf32_to64(as_type<uint>(delta));
        variance64 = metal_sf64_add(variance64, metal_sf64_mul(delta64, delta64));
    }

    sf64Reduction[threadIndex] = variance64;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = normalizationThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            sf64Reduction[threadIndex] = metal_sf64_add(
                sf64Reduction[threadIndex],
                sf64Reduction[threadIndex + stride]
            );
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (threadIndex == 0) {
        rowStats[row * 2] = mean;
        float varianceSum = as_type<float>(metal_sf64_to32(sf64Reduction[0]));
        rowStats[row * 2 + 1] = metal_norm_inv_std_dev_f32(varianceSum, groupSize);
    }
}

kernel void groupnorm_stats_float32(
    device const float* input [[buffer(0)]],
    device float* rowStats [[buffer(1)]],
    constant uint& channels [[buffer(2)]],
    constant uint& spatial [[buffer(3)]],
    constant uint& groups [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_position_in_threadgroup]]
) {
    threadgroup float reduction[normalizationThreadCount];
    threadgroup ulong sf64Reduction[normalizationThreadCount];
    groupnorm_stats_rows_f32(
        input,
        rowStats,
        reduction,
        sf64Reduction,
        channels,
        spatial,
        groups,
        row,
        threadIndex
    );
}
