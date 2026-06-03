#include <metal_stdlib>
#include "../internal/norm_stats.metalinc"

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

    for (uint elementOffset = threadIndex; elementOffset < groupSize; elementOffset += groupnormApplyThreadCount) {
        uint channelInGroup = elementOffset / spatial;
        uint spatialIndex = elementOffset % spatial;
        uint channel = channelStart + channelInGroup;
        float inputValue = input[groupOffset + elementOffset];
        float normalized = (inputValue - mean) * invStdDev;
        float scaleValue = scale[channel];
        float biasValue = bias[channel];
        uint blockStart = spatialIndex & ~3u;

        if (spatial >= 4 && (spatial - blockStart) >= 4) {
            out[groupOffset + elementOffset] = fma(scaleValue, normalized, biasValue);

            continue;
        }

        float scaled = normalized * scaleValue;
        out[groupOffset + elementOffset] = scaled + biasValue;
    }
}
