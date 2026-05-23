#include <metal_stdlib>
#include "attention.metal"

using namespace metal;

FLASH_ATTENTION_KERNEL(flash_attention_float32, Float32TransformerStorage, float)
FLASH_ATTENTION_KERNEL(flash_attention_float16, Float16TransformerStorage, half)
FLASH_ATTENTION_KERNEL(flash_attention_bfloat16, BFloat16TransformerStorage, ushort)
