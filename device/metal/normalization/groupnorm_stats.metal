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

        sf64Reduction[0] = sum64;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (threadIndex == 0) {
        mean = metal_norm_mean_f32(sf64Reduction[0], groupSize);
        reduction[0] = mean;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    mean = reduction[0];

    if (threadIndex == 0) {
#pragma METAL fp_contract(off)
        float lane0 = 0.0f;
        float lane1 = 0.0f;
        float lane2 = 0.0f;
        float lane3 = 0.0f;

        for (uint block = 0; block + 4 <= groupSize; block += 4) {
            float delta0 = input[groupOffset + block] - mean;
            float delta1 = input[groupOffset + block + 1] - mean;
            float delta2 = input[groupOffset + block + 2] - mean;
            float delta3 = input[groupOffset + block + 3] - mean;
            lane0 = fma(delta0, delta0, lane0);
            lane1 = fma(delta1, delta1, lane1);
            lane2 = fma(delta2, delta2, lane2);
            lane3 = fma(delta3, delta3, lane3);
        }

        uint tailStart = groupSize & ~3u;
        uint tailCount = groupSize & 3u;

        if (tailCount >= 1) {
            float delta = input[groupOffset + tailStart] - mean;
            lane0 = fma(delta, delta, lane0);
        }

        if (tailCount >= 2) {
            float delta = input[groupOffset + tailStart + 1] - mean;
            lane1 = fma(delta, delta, lane1);
        }

        if (tailCount >= 3) {
            float delta = input[groupOffset + tailStart + 2] - mean;
            lane2 = fma(delta, delta, lane2);
        }

        reduction[0] = (lane0 + lane1) + (lane2 + lane3);
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
