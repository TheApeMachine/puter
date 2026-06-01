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
    float mean;

    if (threadIndex == 0) {
        ulong sum64 = SF64_NORM_ZERO;

        for (uint offset = 0; offset < groupSize; offset++) {
            float value = input[groupOffset + offset];
            sum64 = metal_sf64_add(sum64, metal_sf32_to64(as_type<uint>(value)));
        }

        mean = metal_norm_mean_f32(sum64, groupSize);
        reduction[0] = mean;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    mean = reduction[0];

    if (threadIndex == 0) {
#pragma METAL fp_contract(off)
        ulong varianceSum64 = SF64_NORM_ZERO;

        for (uint offset = 0; offset < groupSize; offset++) {
            float delta = input[groupOffset + offset] - mean;
            float squared = delta * delta;
            varianceSum64 = metal_sf64_add(
                varianceSum64,
                metal_sf32_to64(as_type<uint>(squared))
            );
        }

        float varianceSum = as_type<float>(metal_sf64_to32(varianceSum64));
        reduction[0] = varianceSum;
#pragma METAL fp_contract(on)
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (threadIndex == 0) {
        rowStats[row * 2] = mean;
        rowStats[row * 2 + 1] = metal_norm_inv_std_dev_f32(reduction[0], groupSize);
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
