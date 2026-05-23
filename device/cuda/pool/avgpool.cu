#include "pool.cuh"

POOL2D_KERNEL_F32(avg_pool2d, pool2d_avg_float32)
POOL2D_KERNEL_F16(avg_pool2d, pool2d_avg_float16)
POOL2D_KERNEL_BF16(avg_pool2d, pool2d_avg_bfloat16)
