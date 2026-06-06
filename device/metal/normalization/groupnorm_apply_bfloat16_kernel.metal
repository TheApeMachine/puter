#include <metal_stdlib>
#include "../internal/norm_stats.metalinc"
#include "../layernorm/layernorm_common.metalinc"

using namespace metal;

constant uint groupnormApplyThreadCount = 256;

kernel void groupnorm_apply_bfloat16(
    device const ushort* input [[buffer(0)]],
    device const ushort* scale [[buffer(1)]],
    device const ushort* bias [[buffer(2)]],
    device ushort* out [[buffer(3)]],
    device const float* rowStats [[buffer(4)]],
    constant uint& channels [[buffer(5)]],
    constant uint& spatial [[buffer(6)]],
    constant uint& groups [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_position_in_threadgroup]]
) {
    uint groupIndex = row % groups;
    uint batchIndex = row / groups;
    uint channelsPerGroup = channels / groups;
    uint channelStart = groupIndex * channelsPerGroup;
    uint groupOffset = (batchIndex * channels + channelStart) * spatial;
    float mean = rowStats[row * 2];
    float invStdDev = rowStats[row * 2 + 1];
    uint vectorStride = groupnormApplyThreadCount * 4;

    for (uint channelInGroup = 0; channelInGroup < channelsPerGroup; channelInGroup++) {
        uint channel = channelStart + channelInGroup;
        uint rowOffset = groupOffset + channelInGroup * spatial;
        float scaleValue = BFloat16NormStorage::load(scale, channel);
        float biasValue = BFloat16NormStorage::load(bias, channel);

        for (uint col = threadIndex * 4; col < spatial; col += vectorStride) {
            for (uint offset = 0; offset < 4; offset++) {
                uint colIndex = col + offset;

                if (colIndex >= spatial) {
                    break;
                }

                float inputValue = BFloat16NormStorage::load(input, rowOffset + colIndex);
                float result = metal_norm_apply_lane_f32_go(
                    inputValue,
                    mean,
                    invStdDev,
                    scaleValue,
                    biasValue
                );
                BFloat16NormStorage::store(out, rowOffset + colIndex, result);
            }
        }
    }
}
