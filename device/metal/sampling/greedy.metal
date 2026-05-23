#include "sampling.metal"

using namespace metal;

GREEDY_SAMPLE_KERNEL(greedy_sample_float32, Float32SamplingStorage, float)
GREEDY_SAMPLE_KERNEL(greedy_sample_float16, Float16SamplingStorage, half)
GREEDY_SAMPLE_KERNEL(greedy_sample_bfloat16, BFloat16SamplingStorage, ushort)
