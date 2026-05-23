#include <metal_stdlib>
#include "../attention/attention.metal"

using namespace metal;

ROPE_KERNEL(rope_float32, Float32TransformerStorage, float)
ROPE_KERNEL(rope_float16, Float16TransformerStorage, half)
ROPE_KERNEL(rope_bfloat16, BFloat16TransformerStorage, ushort)

MULTI_AXIS_ROPE_KERNEL(multi_axis_rope_float32, Float32TransformerStorage, float)
MULTI_AXIS_ROPE_KERNEL(multi_axis_rope_float16, Float16TransformerStorage, half)
MULTI_AXIS_ROPE_KERNEL(multi_axis_rope_bfloat16, BFloat16TransformerStorage, ushort)
