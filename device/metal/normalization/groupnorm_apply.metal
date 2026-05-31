#include <metal_stdlib>

#pragma METAL fp_contract(off)

using namespace metal;

constant uint groupnormApplyThreadCount = 256;

kernel void groupnorm_apply_float32(
    device const float* input [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
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
    uint groupSize = channelsPerGroup * spatial;
    uint groupOffset = (batchIndex * channels + channelStart) * spatial;
    float mean = rowStats[row * 2];
    float invStdDev = rowStats[row * 2 + 1];

    for (uint offset = threadIndex; offset < groupSize; offset += groupnormApplyThreadCount) {
        uint channel = channelStart + offset / spatial;
        float inputValue = input[groupOffset + offset];
        float normalized = inputValue - mean;
        normalized = normalized * invStdDev;
        out[groupOffset + offset] = fma(normalized, scale[channel], bias[channel]);
    }
}
