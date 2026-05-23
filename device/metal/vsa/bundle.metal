#include "vsa.metal"

using namespace metal;

RESEARCH_BINARY_KERNEL(vsa_bundle_float32, vsa_bundle_kernel, Float32ResearchStorage, float)
RESEARCH_BINARY_KERNEL(vsa_bundle_float16, vsa_bundle_kernel, Float16ResearchStorage, half)
RESEARCH_BINARY_KERNEL(vsa_bundle_bfloat16, vsa_bundle_kernel, BFloat16ResearchStorage, ushort)
