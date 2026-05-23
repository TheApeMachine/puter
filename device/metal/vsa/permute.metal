#include "vsa.metal"

using namespace metal;

RESEARCH_UNARY_KERNEL(vsa_permute_float32, vsa_permute_kernel, Float32ResearchStorage, float)
RESEARCH_UNARY_KERNEL(
    vsa_inverse_permute_float32,
    vsa_inverse_permute_kernel,
    Float32ResearchStorage,
    float
)
RESEARCH_UNARY_KERNEL(vsa_permute_float16, vsa_permute_kernel, Float16ResearchStorage, half)
RESEARCH_UNARY_KERNEL(
    vsa_inverse_permute_float16,
    vsa_inverse_permute_kernel,
    Float16ResearchStorage,
    half
)
RESEARCH_UNARY_KERNEL(vsa_permute_bfloat16, vsa_permute_kernel, BFloat16ResearchStorage, ushort)
RESEARCH_UNARY_KERNEL(
    vsa_inverse_permute_bfloat16,
    vsa_inverse_permute_kernel,
    BFloat16ResearchStorage,
    ushort
)
