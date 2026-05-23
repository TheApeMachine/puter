#include <metal_stdlib>
#include "../attention/attention.metal"

using namespace metal;

EMBEDDING_LOOKUP_KERNEL(embedding_lookup_float32, Float32TransformerStorage, float)
EMBEDDING_LOOKUP_KERNEL(embedding_lookup_float16, Float16TransformerStorage, half)
EMBEDDING_LOOKUP_KERNEL(embedding_lookup_bfloat16, BFloat16TransformerStorage, ushort)
