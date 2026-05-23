#include "normalization.metal"

using namespace metal;

INSTANCENORM_KERNEL(instancenorm_float32, Float32NormStorage, float)
INSTANCENORM_KERNEL(instancenorm_float16, Float16NormStorage, half)
INSTANCENORM_KERNEL(instancenorm_bfloat16, BFloat16NormStorage, ushort)
