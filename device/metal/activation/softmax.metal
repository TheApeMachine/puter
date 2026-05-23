#include <metal_stdlib>
#include "activation_bf16_math.metalinc"

using namespace metal;

constant uint softmaxThreadCount = 256;
constant uint softmaxSimdgroupWidth = 32;

kernel void softmax_float32(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_position_in_threadgroup]]
) {
    threadgroup float reduction[256];
    uint rowOffset = row * cols;
    float localMax = -3.4028234663852886e38f;

    for (uint col = threadIndex; col < cols; col += softmaxThreadCount) {
        localMax = max(localMax, input[rowOffset + col]);
    }

    localMax = simd_max(localMax);

    uint simdLane = threadIndex & (softmaxSimdgroupWidth - 1u);
    uint simdgroupIndex = threadIndex / softmaxSimdgroupWidth;

    if (simdLane == 0) {
        reduction[simdgroupIndex] = localMax;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (threadIndex == 0) {
        float rowMax = reduction[0];
        uint simdgroupCount = softmaxThreadCount / softmaxSimdgroupWidth;

        for (uint groupIndex = 1; groupIndex < simdgroupCount; groupIndex++) {
            rowMax = max(rowMax, reduction[groupIndex]);
        }

        reduction[0] = rowMax;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum = reduction[0];
    float localSum = 0.0f;

    for (uint col = threadIndex; col < cols; col += softmaxThreadCount) {
        localSum += exp(input[rowOffset + col] - maximum);
    }

    localSum = simd_sum(localSum);

    if (simdLane == 0) {
        reduction[simdgroupIndex] = localSum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (threadIndex == 0) {
        float rowSum = reduction[0];
        uint simdgroupCount = softmaxThreadCount / softmaxSimdgroupWidth;

        for (uint groupIndex = 1; groupIndex < simdgroupCount; groupIndex++) {
            rowSum += reduction[groupIndex];
        }

        reduction[0] = rowSum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = reduction[0];

    for (uint col = threadIndex; col < cols; col += softmaxThreadCount) {
        float value = sum == 0.0f ? 0.0f : exp(input[rowOffset + col] - maximum) / sum;
        out[rowOffset + col] = value;
    }
}

kernel void softmax_float16(
    device const half* input [[buffer(0)]],
    device half* out [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_position_in_threadgroup]]
) {
    threadgroup half reduction[256];
    uint rowOffset = row * cols;
    half localMax = half(-65504.0h);

    for (uint col = threadIndex; col < cols; col += softmaxThreadCount) {
        localMax = max(localMax, input[rowOffset + col]);
    }

    reduction[threadIndex] = localMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (threadIndex == 0) {
        half rowMax = reduction[0];

        for (uint offset = 1; offset < softmaxThreadCount; offset++) {
            rowMax = max(rowMax, reduction[offset]);
        }

        reduction[0] = rowMax;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    half maximum = reduction[0];
    half localSum = half(0.0h);

    for (uint col = threadIndex; col < cols; col += softmaxThreadCount) {
        localSum = localSum + exp(input[rowOffset + col] - maximum);
    }

    reduction[threadIndex] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (threadIndex == 0) {
        half rowSum = reduction[0];

        for (uint offset = 1; offset < softmaxThreadCount; offset++) {
            rowSum = rowSum + reduction[offset];
        }

        reduction[0] = rowSum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    half sum = reduction[0];

    for (uint col = threadIndex; col < cols; col += softmaxThreadCount) {
        half numerator = exp(input[rowOffset + col] - maximum);
        out[rowOffset + col] = numerator / sum;
    }
}

kernel void softmax_bfloat16(
    device const ushort* input [[buffer(0)]],
    device ushort* out [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_position_in_threadgroup]]
) {
    threadgroup bfloat reduction[256];
    uint rowOffset = row * cols;
    bfloat localMax = bfloat(-3.38953139e38);

    for (uint col = threadIndex; col < cols; col += softmaxThreadCount) {
        bfloat loaded = as_type<bfloat>(input[rowOffset + col]);
        localMax = activation_bf16_max(localMax, loaded);
    }

    reduction[threadIndex] = localMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (threadIndex == 0) {
        bfloat rowMax = reduction[0];

        for (uint offset = 1; offset < softmaxThreadCount; offset++) {
            rowMax = activation_bf16_max(rowMax, reduction[offset]);
        }

        reduction[0] = rowMax;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    bfloat maximum = reduction[0];
    bfloat localSum = activation_bf16_zero();

    for (uint col = threadIndex; col < cols; col += softmaxThreadCount) {
        bfloat loaded = as_type<bfloat>(input[rowOffset + col]);
        localSum = localSum + activation_bf16_exp(loaded - maximum);
    }

    reduction[threadIndex] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (threadIndex == 0) {
        bfloat rowSum = reduction[0];

        for (uint offset = 1; offset < softmaxThreadCount; offset++) {
            rowSum = rowSum + reduction[offset];
        }

        reduction[0] = rowSum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    bfloat sum = reduction[0];

    for (uint col = threadIndex; col < cols; col += softmaxThreadCount) {
        bfloat loaded = as_type<bfloat>(input[rowOffset + col]);
        bfloat numerator = activation_bf16_exp(loaded - maximum);
        out[rowOffset + col] = as_type<ushort>(numerator / sum);
    }
}
