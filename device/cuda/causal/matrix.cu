#include "matrix.cuh"

CAUSAL_CHOLESKY_KERNEL(cholesky_float32, float, causal_load_f32, causal_store_f32)
CAUSAL_CHOLESKY_KERNEL(cholesky_float16, __half, causal_load_f16, causal_store_f16)
CAUSAL_CHOLESKY_KERNEL(cholesky_bfloat16, __nv_bfloat16, causal_load_bf16, causal_store_bf16)
