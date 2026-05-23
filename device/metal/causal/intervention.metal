#include <metal_stdlib>
#include "causal.metal"

using namespace metal;

DO_INTERVENE_KERNEL(do_intervene_float32, Float32CausalStorage, float)
DO_INTERVENE_KERNEL(do_intervene_float16, Float16CausalStorage, half)
DO_INTERVENE_KERNEL(do_intervene_bfloat16, BFloat16CausalStorage, ushort)
CATE_KERNEL(cate_float32, Float32CausalStorage, float)
CATE_KERNEL(cate_float16, Float16CausalStorage, half)
CATE_KERNEL(cate_bfloat16, BFloat16CausalStorage, ushort)
COUNTERFACTUAL_KERNEL(counterfactual_float32, Float32CausalStorage, float)
COUNTERFACTUAL_KERNEL(counterfactual_float16, Float16CausalStorage, half)
COUNTERFACTUAL_KERNEL(counterfactual_bfloat16, BFloat16CausalStorage, ushort)
IV_KERNELS(iv_estimate_float32, Float32CausalStorage, float)
IV_KERNELS(iv_estimate_float16, Float16CausalStorage, half)
IV_KERNELS(iv_estimate_bfloat16, BFloat16CausalStorage, ushort)
