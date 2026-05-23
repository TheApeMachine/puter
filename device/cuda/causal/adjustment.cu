#include "causal.cuh"

CAUSAL_BACKDOOR_KERNEL(backdoor_adjustment_float32, float, causal_load_f32, causal_store_f32)
CAUSAL_BACKDOOR_KERNEL(backdoor_adjustment_float16, __half, causal_load_f16, causal_store_f16)
CAUSAL_BACKDOOR_KERNEL(backdoor_adjustment_bfloat16, __nv_bfloat16, causal_load_bf16, causal_store_bf16)

CAUSAL_FRONTDOOR_KERNEL(frontdoor_adjustment_float32, float, causal_load_f32, causal_store_f32)
CAUSAL_FRONTDOOR_KERNEL(frontdoor_adjustment_float16, __half, causal_load_f16, causal_store_f16)
CAUSAL_FRONTDOOR_KERNEL(frontdoor_adjustment_bfloat16, __nv_bfloat16, causal_load_bf16, causal_store_bf16)
