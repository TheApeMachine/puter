#include <metal_stdlib>
#include "causal.metal"

using namespace metal;

BACKDOOR_KERNEL(backdoor_adjustment_float32, Float32CausalStorage, float)
BACKDOOR_KERNEL(backdoor_adjustment_float16, Float16CausalStorage, half)
BACKDOOR_KERNEL(backdoor_adjustment_bfloat16, BFloat16CausalStorage, ushort)
FRONTDOOR_KERNEL(frontdoor_adjustment_float32, Float32CausalStorage, float)
FRONTDOOR_KERNEL(frontdoor_adjustment_float16, Float16CausalStorage, half)
FRONTDOOR_KERNEL(frontdoor_adjustment_bfloat16, BFloat16CausalStorage, ushort)
