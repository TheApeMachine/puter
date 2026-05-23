#include <metal_stdlib>
#include "../attention/attention.metal"

using namespace metal;

EMBEDDING_BAG_KERNEL(embedding_bag_float32, Float32TransformerStorage, float)
EMBEDDING_BAG_KERNEL(embedding_bag_float16, Float16TransformerStorage, half)
EMBEDDING_BAG_KERNEL(embedding_bag_bfloat16, BFloat16TransformerStorage, ushort)
