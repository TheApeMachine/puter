#include "losses.metal"

using namespace metal;

#define PAIR_LOSS_KERNEL(name, storage, scalar, op) \
#define CROSS_ENTROPY_KERNEL(name, storage, scalar) \
#define CROSS_ENTROPY_KERNEL(name, storage, scalar) \
    cross_entropy_loss_partial<storage, scalar>( \
PAIR_LOSS_KERNEL(binary_cross_entropy_float32, Float32LossStorage, float, BinaryCrossEntropyLossOp)
CROSS_ENTROPY_KERNEL(cross_entropy_float32, Float32LossStorage, float)
PAIR_LOSS_KERNEL(binary_cross_entropy_float16, Float16LossStorage, half, BinaryCrossEntropyLossOp)
CROSS_ENTROPY_KERNEL(cross_entropy_float16, Float16LossStorage, half)
PAIR_LOSS_KERNEL(binary_cross_entropy_bfloat16, BFloat16LossStorage, ushort, BinaryCrossEntropyLossOp)
CROSS_ENTROPY_KERNEL(cross_entropy_bfloat16, BFloat16LossStorage, ushort)
