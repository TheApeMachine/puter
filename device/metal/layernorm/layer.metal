#include "layernorm.metal"

using namespace metal;

#define LAYERNORM_KERNEL(name, storage, scalar) \
#define RMSNORM_KERNEL(name, storage, scalar) \
#define ADAPTIVE_RMSNORM_KERNEL(name, storage, scalar) \
#define MODULATED_LAYERNORM_KERNEL(name, storage, scalar) \
#define GATED_RESIDUAL_KERNEL(name, storage, scalar) \
LAYERNORM_KERNEL(layernorm_float16, Float16NormStorage, half)
LAYERNORM_KERNEL(layernorm_bfloat16, BFloat16NormStorage, ushort)
RMSNORM_KERNEL(rmsnorm_float32, Float32NormStorage, float)
RMSNORM_KERNEL(rmsnorm_float16, Float16NormStorage, half)
RMSNORM_KERNEL(rmsnorm_bfloat16, BFloat16NormStorage, ushort)
ADAPTIVE_RMSNORM_KERNEL(adaptive_rmsnorm_float32, Float32NormStorage, float)
ADAPTIVE_RMSNORM_KERNEL(adaptive_rmsnorm_float16, Float16NormStorage, half)
ADAPTIVE_RMSNORM_KERNEL(adaptive_rmsnorm_bfloat16, BFloat16NormStorage, ushort)
MODULATED_LAYERNORM_KERNEL(modulated_layernorm_float32, Float32NormStorage, float)
MODULATED_LAYERNORM_KERNEL(modulated_layernorm_float16, Float16NormStorage, half)
MODULATED_LAYERNORM_KERNEL(modulated_layernorm_bfloat16, BFloat16NormStorage, ushort)
GATED_RESIDUAL_KERNEL(gated_residual_float32, Float32NormStorage, float)
GATED_RESIDUAL_KERNEL(gated_residual_float16, Float16NormStorage, half)
GATED_RESIDUAL_KERNEL(gated_residual_bfloat16, BFloat16NormStorage, ushort)
static inline void layernorm_rows(
static inline void rmsnorm_rows(
static inline void adaptive_rmsnorm_rows(
static inline void modulated_layernorm_rows(
static inline void gated_residual_values(
    layernorm_rows<storage, scalar>(input, scale, bias, out, reduction, cols, row, threadIndex); \
    rmsnorm_rows<storage, scalar>(input, scale, out, reduction, cols, epsilon, row, threadIndex); \
    adaptive_rmsnorm_rows<storage, scalar>(input, modulation, out, reduction, cols, row, threadIndex); \
    modulated_layernorm_rows<storage, scalar>( \
    gated_residual_values<storage, scalar>( \
kernel void layernorm_float32(
LAYERNORM_KERNEL(layernorm_float16, Float16NormStorage, half)
LAYERNORM_KERNEL(layernorm_bfloat16, BFloat16NormStorage, ushort)
RMSNORM_KERNEL(rmsnorm_float32, Float32NormStorage, float)
RMSNORM_KERNEL(rmsnorm_float16, Float16NormStorage, half)
RMSNORM_KERNEL(rmsnorm_bfloat16, BFloat16NormStorage, ushort)
ADAPTIVE_RMSNORM_KERNEL(adaptive_rmsnorm_float32, Float32NormStorage, float)
ADAPTIVE_RMSNORM_KERNEL(adaptive_rmsnorm_float16, Float16NormStorage, half)
ADAPTIVE_RMSNORM_KERNEL(adaptive_rmsnorm_bfloat16, BFloat16NormStorage, ushort)
MODULATED_LAYERNORM_KERNEL(modulated_layernorm_float32, Float32NormStorage, float)
MODULATED_LAYERNORM_KERNEL(modulated_layernorm_float16, Float16NormStorage, half)
MODULATED_LAYERNORM_KERNEL(modulated_layernorm_bfloat16, BFloat16NormStorage, ushort)
GATED_RESIDUAL_KERNEL(gated_residual_float32, Float32NormStorage, float)
GATED_RESIDUAL_KERNEL(gated_residual_float16, Float16NormStorage, half)
GATED_RESIDUAL_KERNEL(gated_residual_bfloat16, BFloat16NormStorage, ushort)
