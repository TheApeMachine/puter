#include "normalization.metal"

using namespace metal;

#define BATCHNORM_DENORM_KERNEL(name, storage, scalar) \
#define BATCHNORM_EVAL_KERNEL(name, storage, scalar) \
BATCHNORM_DENORM_KERNEL(batchnorm_denorm_float32, Float32NormStorage, float)
BATCHNORM_DENORM_KERNEL(batchnorm_denorm_float16, Float16NormStorage, half)
BATCHNORM_DENORM_KERNEL(batchnorm_denorm_bfloat16, BFloat16NormStorage, ushort)
BATCHNORM_EVAL_KERNEL(batchnorm_eval_float32, Float32NormStorage, float)
BATCHNORM_EVAL_KERNEL(batchnorm_eval_float16, Float16NormStorage, half)
BATCHNORM_EVAL_KERNEL(batchnorm_eval_bfloat16, BFloat16NormStorage, ushort)
static inline void batchnorm_denorm_values(
static inline void batchnorm_eval_rows(
#define BATCHNORM_DENORM_KERNEL(name, storage, scalar) \
    batchnorm_denorm_values<storage, scalar>(input, mean, variance, out, channels, spatial, row, threadIndex); \
#define BATCHNORM_EVAL_KERNEL(name, storage, scalar) \
    batchnorm_eval_rows<storage, scalar>( \
BATCHNORM_DENORM_KERNEL(batchnorm_denorm_float32, Float32NormStorage, float)
BATCHNORM_DENORM_KERNEL(batchnorm_denorm_float16, Float16NormStorage, half)
BATCHNORM_DENORM_KERNEL(batchnorm_denorm_bfloat16, BFloat16NormStorage, ushort)
BATCHNORM_EVAL_KERNEL(batchnorm_eval_float32, Float32NormStorage, float)
BATCHNORM_EVAL_KERNEL(batchnorm_eval_float16, Float16NormStorage, half)
BATCHNORM_EVAL_KERNEL(batchnorm_eval_bfloat16, BFloat16NormStorage, ushort)
