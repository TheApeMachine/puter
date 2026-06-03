#include <metal_stdlib>
#include "../internal/norm_stats.metalinc"
#include "../layernorm/layernorm_common.metalinc"

using namespace metal;

constant uint groupnormApplyThreadCount = 256;

kernel void groupnorm_apply_float16(
    device const half* input [[buffer(0)]],
    device const half* scale [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* out [[buffer(3)]],
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
        float scaleValue = Float16NormStorage::load(scale, channel);
        float biasValue = Float16NormStorage::load(bias, channel);

        for (uint col = threadIndex * 4; col < spatial; col += vectorStride) {
            for (uint offset = 0; offset < 4; offset++) {
                uint colIndex = col + offset;

                if (colIndex >= spatial) {
                    break;
                }

#pragma METAL fp_contract(off)
                float inputValue = Float16NormStorage::load(input, rowOffset + colIndex);
                float normalized = (inputValue - mean) * invStdDev;
                float scaled = normalized * scaleValue;
                float result = scaled + biasValue;
#pragma METAL fp_contract(on)
                Float16NormStorage::store(out, rowOffset + colIndex, result);
            }
        }
    }
}
