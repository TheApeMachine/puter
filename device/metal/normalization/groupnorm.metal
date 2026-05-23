#include "normalization.metal"

using namespace metal;

GROUPNORM_KERNEL(groupnorm_float32, Float32NormStorage, float)
GROUPNORM_KERNEL(groupnorm_float16, Float16NormStorage, half)
GROUPNORM_KERNEL(groupnorm_bfloat16, BFloat16NormStorage, ushort)
