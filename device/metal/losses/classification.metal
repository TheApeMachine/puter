#include "losses.metal"

using namespace metal;

PAIR_LOSS_KERNEL(binary_cross_entropy_float32, Float32LossStorage, float, BinaryCrossEntropyLossOp)
CROSS_ENTROPY_KERNEL(cross_entropy_float32, Float32LossStorage, float)
PAIR_LOSS_KERNEL(binary_cross_entropy_float16, Float16LossStorage, half, BinaryCrossEntropyLossOp)
CROSS_ENTROPY_KERNEL(cross_entropy_float16, Float16LossStorage, half)
PAIR_LOSS_KERNEL(binary_cross_entropy_bfloat16, BFloat16LossStorage, ushort, BinaryCrossEntropyLossOp)
CROSS_ENTROPY_KERNEL(cross_entropy_bfloat16, BFloat16LossStorage, ushort)
