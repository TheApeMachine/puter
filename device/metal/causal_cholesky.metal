#include <metal_stdlib>

using namespace metal;

kernel void cholesky_float32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& order [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) {
        return;
    }

    for (uint row = 0; row < order; row++) {
        for (uint col = 0; col <= row; col++) {
            float sum = input[row * order + col];

            for (uint inner = 0; inner < col; inner++) {
                sum -= output[row * order + inner] * output[col * order + inner];
            }

            if (row == col) {
                output[row * order + col] = sqrt(max(sum, 0.0f));
            } else {
                output[row * order + col] = sum / output[col * order + col];
            }
        }
    }
}
