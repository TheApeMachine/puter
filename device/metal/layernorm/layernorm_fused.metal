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

LAYERNORM_KERNEL(layernorm_float32, Float32NormStorage, float)
LAYERNORM_KERNEL(layernorm_float16, Float16NormStorage, half)
LAYERNORM_KERNEL(layernorm_bfloat16, BFloat16NormStorage, ushort)
