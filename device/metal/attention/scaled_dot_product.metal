#include <metal_stdlib>
#include "attention.metal"

using namespace metal;

ATTENTION_SCORES_KERNEL(attention_scores_float32, Float32TransformerStorage, float)
ATTENTION_SCORES_KERNEL(attention_scores_float16, Float16TransformerStorage, half)
ATTENTION_SCORES_KERNEL(attention_scores_bfloat16, BFloat16TransformerStorage, ushort)

kernel void attention_softmax(
    device float* scores [[buffer(0)]],
    constant uint& seqK [[buffer(1)]],
    uint row [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_position_in_threadgroup]]
) {
    threadgroup float reduction[256];
    attention_softmax_row(scores, reduction, seqK, row, threadIndex);
}

ATTENTION_WEIGHTED_KERNEL(attention_weighted_float32, Float32TransformerStorage, float)
ATTENTION_WEIGHTED_KERNEL(attention_weighted_float16, Float16TransformerStorage, half)
ATTENTION_WEIGHTED_KERNEL(attention_weighted_bfloat16, BFloat16TransformerStorage, ushort)
