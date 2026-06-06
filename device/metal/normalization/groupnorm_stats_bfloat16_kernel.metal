#include "../layernorm/layernorm_common.metalinc"
#include "../internal/norm_stats.metalinc"

using namespace metal;

static inline void groupnorm_stats_rows_bf16(
    device const ushort* input,
    device float* rowStats,
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

    if (threadIndex == 0) {
#pragma METAL fp_contract(off)
        ushort sumBf16 = metal_layernorm_sum_bf16_native(input, groupOffset, groupSize);
        float mean = bf16_to_float_norm(sumBf16) / float(groupSize);
        float varianceSum = compute_variance_sum_f32<BFloat16NormStorage, ushort>(
            input,
            groupOffset,
            groupSize,
            mean
        );
        float variance = varianceSum / float(groupSize);
        float invStdDev = metal_norm_inv_std_dev_f32_go_plain(variance + layerNormEpsilonMetal);

        rowStats[row * 2] = mean;
        rowStats[row * 2 + 1] = invStdDev;
#pragma METAL fp_contract(on)
    }
}

kernel void groupnorm_stats_bfloat16(
    device const ushort* input [[buffer(0)]],
    device float* rowStats [[buffer(1)]],
    constant uint& channels [[buffer(2)]],
    constant uint& spatial [[buffer(3)]],
    constant uint& groups [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_position_in_threadgroup]]
) {
    groupnorm_stats_rows_bf16(input, rowStats, channels, spatial, groups, row, threadIndex);
}
