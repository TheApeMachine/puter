#include "normalization.metal"

using namespace metal;

#define INSTANCENORM_KERNEL(name, storage, scalar) \
INSTANCENORM_KERNEL(instancenorm_float16, Float16NormStorage, half)
INSTANCENORM_KERNEL(instancenorm_bfloat16, BFloat16NormStorage, ushort)
static inline void instancenorm_rows(
#define INSTANCENORM_KERNEL(name, storage, scalar) \
    instancenorm_rows<storage, scalar>( \
kernel void instancenorm_float32(
INSTANCENORM_KERNEL(instancenorm_float16, Float16NormStorage, half)
INSTANCENORM_KERNEL(instancenorm_bfloat16, BFloat16NormStorage, ushort)
kernel void instancenorm_stats_float32(
