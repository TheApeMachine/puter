#include "hawkes.metal"

using namespace metal;

#define HAWKES_KERNEL_MATRIX_KERNEL(name, storage, scalar) \
    hawkes_kernel_matrix_kernel<storage, scalar>(events, alpha, beta, out, eventCount, index); \
HAWKES_KERNEL_MATRIX_KERNEL(hawkes_kernel_matrix_float32, Float32HawkesMarkovStorage, float)
HAWKES_KERNEL_MATRIX_KERNEL(hawkes_kernel_matrix_float16, Float16HawkesMarkovStorage, half)
HAWKES_KERNEL_MATRIX_KERNEL(hawkes_kernel_matrix_bfloat16, BFloat16HawkesMarkovStorage, ushort)
kernel void hawkes_exp_float32(
