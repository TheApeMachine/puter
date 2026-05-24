#include "layernorm_common.metalinc"
#include "layernorm_apply.metalinc"

kernel void layernorm_apply_float32(
    device const float* input [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    device const float* rowStats [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_position_in_threadgroup]]
) {
    float mean = rowStats[row * 2];
    float invStdDev = rowStats[row * 2 + 1];
    uint rowOffset = row * cols;

    layernorm_apply_row<Float32NormStorage, float>(
        input,
        scale,
        bias,
        out,
        rowOffset,
        cols,
        mean,
        invStdDev,
        threadIndex
    );
}
