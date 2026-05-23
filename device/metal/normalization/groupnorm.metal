#include "normalization.metal"

using namespace metal;

#define GROUPNORM_KERNEL(name, storage, scalar) \
GROUPNORM_KERNEL(groupnorm_float16, Float16NormStorage, half)
GROUPNORM_KERNEL(groupnorm_bfloat16, BFloat16NormStorage, ushort)
static inline void groupnorm_rows(
#define GROUPNORM_KERNEL(name, storage, scalar) \
    groupnorm_rows<storage, scalar>( \
kernel void groupnorm_float32(
GROUPNORM_KERNEL(groupnorm_float16, Float16NormStorage, half)
GROUPNORM_KERNEL(groupnorm_bfloat16, BFloat16NormStorage, ushort)
kernel void groupnorm_stats_float32(
