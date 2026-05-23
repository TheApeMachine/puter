#include "sampling.metal"

using namespace metal;

kernel void sampling_bitonic_stepkernel void sampling_bitonic_step(
kernel void sampling_draw_sorted(
#define SAMPLING_INIT_KERNEL(name, storage, scalar) \
    sampling_init_kernel<storage, scalar>(logits, scores, indices, count, paddedCount, index); \
SAMPLING_INIT_KERNEL(sampling_init_float32, Float32SamplingStorage, float)
SAMPLING_INIT_KERNEL(sampling_init_float16, Float16SamplingStorage, half)
SAMPLING_INIT_KERNEL(sampling_init_bfloat16, BFloat16SamplingStorage, ushort)
