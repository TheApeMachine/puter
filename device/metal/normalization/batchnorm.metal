#include "normalization.metal"

using namespace metal;

BATCHNORM_DENORM_KERNEL(batchnorm_denorm_float32, Float32NormStorage, float)
BATCHNORM_DENORM_KERNEL(batchnorm_denorm_float16, Float16NormStorage, half)
BATCHNORM_DENORM_KERNEL(batchnorm_denorm_bfloat16, BFloat16NormStorage, ushort)
BATCHNORM_EVAL_KERNEL(batchnorm_eval_float32, Float32NormStorage, float)
BATCHNORM_EVAL_KERNEL(batchnorm_eval_float16, Float16NormStorage, half)
BATCHNORM_EVAL_KERNEL(batchnorm_eval_bfloat16, BFloat16NormStorage, ushort)
