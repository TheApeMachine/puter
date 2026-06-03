#include "../internal/norm_stats.metalinc"
#include "../layernorm/layernorm_thread_count.h"

using namespace metal;

constant uint normalizationThreadCount = 256;

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

    ulong sum64 = metal_norm_reduce_sum_sf64(
        input,
        sf64Reduction,
        groupOffset,
        groupSize,
        threadIndex,
        normalizationThreadCount
    );

    if (threadIndex == 0) {
        float sumF32 = as_type<float>(metal_sf64_to32(sum64));
        float mean = as_type<float>(
            metal_sf64_to32(
                metal_sf64_div(
                    metal_sf32_to64(as_type<uint>(sumF32)),
                    metal_sf64_int_to64(int(groupSize))
                )
            )
        );
        float varianceSum = metal_norm_squared_diff_sum_double_f32(
            input,
            groupOffset,
            groupSize,
            mean
        );

        float invStdDev = metal_norm_inv_std_dev_f32(varianceSum, groupSize);

        rowStats[row * 2] = mean;
        rowStats[row * 2 + 1] = invStdDev;
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
