#include "vsa.metal"

using namespace metal;

RESEARCH_BINARY_KERNEL(vsa_bind_float32, vsa_bind_kernel, Float32ResearchStorage, float)
RESEARCH_BINARY_KERNEL(vsa_bind_float16, vsa_bind_kernel, Float16ResearchStorage, half)
RESEARCH_BINARY_KERNEL(vsa_bind_bfloat16, vsa_bind_kernel, BFloat16ResearchStorage, ushort)
