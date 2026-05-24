#include "layernorm.metal"

using namespace metal;

#define LAYERNORM_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device const scalar* scale [[buffer(1)]], \
    device const scalar* bias [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    constant uint& cols [[buffer(4)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    threadgroup float reduction[normalizationThreadCount]; \
    threadgroup ulong sf64Reduction[normalizationThreadCount]; \
    layernorm_rows<storage, scalar>( \
        input, scale, bias, out, reduction, sf64Reduction, cols, row, threadIndex \
    ); \
}

#define RMSNORM_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device const scalar* scale [[buffer(1)]], \
    device scalar* out [[buffer(2)]], \
    constant uint& cols [[buffer(3)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    threadgroup float reduction[256]; \
    rmsnorm_rows<storage, scalar>(input, scale, out, reduction, cols, row, threadIndex); \
}

#define ADAPTIVE_RMSNORM_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device const scalar* modulation [[buffer(1)]], \
    device scalar* out [[buffer(2)]], \
    constant uint& cols [[buffer(3)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    threadgroup float reduction[256]; \
    adaptive_rmsnorm_rows<storage, scalar>(input, modulation, out, reduction, cols, row, threadIndex); \
}

#define MODULATED_LAYERNORM_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device const scalar* modulation [[buffer(1)]], \
    device scalar* out [[buffer(2)]], \
    constant uint& cols [[buffer(3)]], \
    constant uint& rowsPerBatch [[buffer(4)]], \
    constant uint& modulationCols [[buffer(5)]], \
    constant uint& modulationSet [[buffer(6)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    threadgroup float reduction[256]; \
    modulated_layernorm_rows<storage, scalar>( \
        input, modulation, out, reduction, cols, rowsPerBatch, modulationCols, modulationSet, row, threadIndex \
    ); \
}

#define GATED_RESIDUAL_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* residual [[buffer(0)]], \
    device const scalar* branch [[buffer(1)]], \
    device const scalar* modulation [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    constant uint& cols [[buffer(4)]], \
    constant uint& rowsPerBatch [[buffer(5)]], \
    constant uint& modulationCols [[buffer(6)]], \
    constant uint& modulationSet [[buffer(7)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    gated_residual_values<storage, scalar>( \
        residual, branch, modulation, out, cols, rowsPerBatch, modulationCols, modulationSet, row, threadIndex \
    ); \
}

LAYERNORM_KERNEL(layernorm_float32, Float32NormStorage, float)
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
