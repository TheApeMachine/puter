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
    float4 meanVec = float4(mean);
    float4 invStdDevVec = float4(invStdDev);
    uint vectorStride = groupnormApplyThreadCount * 4;

    for (uint offset = threadIndex * 4; offset < groupSize; offset += vectorStride) {
        if (offset + 4 <= groupSize) {
            float4 inputValue = float4(
                input[groupOffset + offset],
                input[groupOffset + offset + 1],
                input[groupOffset + offset + 2],
                input[groupOffset + offset + 3]
            );
            float4 scaleValue = float4(
                scale[channelStart + (offset) / spatial],
                scale[channelStart + (offset + 1) / spatial],
                scale[channelStart + (offset + 2) / spatial],
                scale[channelStart + (offset + 3) / spatial]
            );
            float4 biasValue = float4(
                bias[channelStart + (offset) / spatial],
                bias[channelStart + (offset + 1) / spatial],
                bias[channelStart + (offset + 2) / spatial],
                bias[channelStart + (offset + 3) / spatial]
            );
            float4 delta = inputValue - meanVec;
            delta = delta * invStdDevVec;
            float4 result = fma(scaleValue, delta, biasValue);

            out[groupOffset + offset] = result.x;
            out[groupOffset + offset + 1] = result.y;
            out[groupOffset + offset + 2] = result.z;
            out[groupOffset + offset + 3] = result.w;

            continue;
        }

        for (uint lane = 0; lane < 4; lane++) {
            uint elementOffset = offset + lane;

            if (elementOffset >= groupSize) {
                break;
            }

            uint channel = channelStart + elementOffset / spatial;
            float inputValue = input[groupOffset + elementOffset];
            float delta = inputValue - mean;
            delta = delta * invStdDev;
            out[groupOffset + elementOffset] = fma(scale[channel], delta, bias[channel]);
        }
    }
}
