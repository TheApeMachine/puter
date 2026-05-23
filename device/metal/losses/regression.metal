#include "losses.metal"

using namespace metal;

#define PAIR_LOSS_KERNEL(name, storage, scalar, op) \
#define CROSS_ENTROPY_KERNEL(name, storage, scalar) \
#define PAIR_LOSS_KERNEL(name, storage, scalar, op) \
PAIR_LOSS_KERNEL(mse_loss_float32, Float32LossStorage, float, MSELossOp)
PAIR_LOSS_KERNEL(mae_loss_float32, Float32LossStorage, float, MAELossOp)
PAIR_LOSS_KERNEL(huber_loss_float32, Float32LossStorage, float, HuberLossOp)
PAIR_LOSS_KERNEL(kl_divergence_float32, Float32LossStorage, float, KLDivergenceLossOp)
PAIR_LOSS_KERNEL(mse_loss_float16, Float16LossStorage, half, MSELossOp)
PAIR_LOSS_KERNEL(mae_loss_float16, Float16LossStorage, half, MAELossOp)
PAIR_LOSS_KERNEL(huber_loss_float16, Float16LossStorage, half, HuberLossOp)
PAIR_LOSS_KERNEL(kl_divergence_float16, Float16LossStorage, half, KLDivergenceLossOp)
PAIR_LOSS_KERNEL(mse_loss_bfloat16, BFloat16LossStorage, ushort, MSELossOp)
PAIR_LOSS_KERNEL(mae_loss_bfloat16, BFloat16LossStorage, ushort, MAELossOp)
PAIR_LOSS_KERNEL(huber_loss_bfloat16, BFloat16LossStorage, ushort, HuberLossOp)
PAIR_LOSS_KERNEL(kl_divergence_bfloat16, BFloat16LossStorage, ushort, KLDivergenceLossOp)
