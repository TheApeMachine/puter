#include "layernorm_common.metalinc"

using namespace metal;

static inline void layernorm_stats_rows_f32(
    device const float* input,
    device float* rowStats,
    threadgroup float* reduction,
    threadgroup ulong* sf64Reduction,
    constant uint& cols,
    uint row,
    uint threadIndex
) {
    uint rowOffset = row * cols;
    float mean;
    float invStdDev;

    compute_row_stats<Float32NormStorage, float>(
        input,
        reduction,
        sf64Reduction,
        rowOffset,
        cols,
        threadIndex,
        mean,
        invStdDev
    );

    if (threadIndex == 0) {
        rowStats[row * 2] = mean;
        rowStats[row * 2 + 1] = invStdDev;
    }
}

kernel void layernorm_stats_float32(
    device const float* input [[buffer(0)]],
    device float* rowStats [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_position_in_threadgroup]]
) {
    threadgroup float reduction[normalizationThreadCount];
    threadgroup ulong sf64Reduction[normalizationThreadCount];
    layernorm_stats_rows_f32(input, rowStats, reduction, sf64Reduction, cols, row, threadIndex);
}

static inline void layernorm_stats_rows_f16(
    device const half* input,
    device float* rowStats,
    threadgroup float* reduction,
    threadgroup ulong* sf64Reduction,
    constant uint& cols,
    uint row,
    uint threadIndex
) {
    uint rowOffset = row * cols;
    float mean;
    float invStdDev;

    compute_row_stats<Float16NormStorage, half>(
        input,
        reduction,
        sf64Reduction,
        rowOffset,
        cols,
        threadIndex,
        mean,
        invStdDev
    );

    if (threadIndex == 0) {
        rowStats[row * 2] = mean;
        rowStats[row * 2 + 1] = invStdDev;
    }
}

kernel void layernorm_stats_float16(
    device const half* input [[buffer(0)]],
    device float* rowStats [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_position_in_threadgroup]]
) {
    threadgroup float reduction[normalizationThreadCount];
    threadgroup ulong sf64Reduction[normalizationThreadCount];
    layernorm_stats_rows_f16(input, rowStats, reduction, sf64Reduction, cols, row, threadIndex);
}
