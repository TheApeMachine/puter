#include "hawkes.metal"

using namespace metal;

HAWKES_KERNEL_MATRIX_KERNEL(hawkes_kernel_matrix_float32, Float32HawkesMarkovStorage, float)
HAWKES_KERNEL_MATRIX_KERNEL(hawkes_kernel_matrix_float16, Float16HawkesMarkovStorage, half)
HAWKES_KERNEL_MATRIX_KERNEL(hawkes_kernel_matrix_bfloat16, BFloat16HawkesMarkovStorage, ushort)

kernel void hawkes_exp_float32(
    device const float* inputVector [[buffer(0)]],
    device float* outVector [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) {
        return;
    }

    outVector[index] = metal_hawkes_exp32(inputVector[index]);
}
