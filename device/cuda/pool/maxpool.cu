#include "pool.cuh"

POOL2D_KERNEL_F32(max_pool2d, pool2d_max_float32)
POOL2D_KERNEL_F16(max_pool2d, pool2d_max_float16)
POOL2D_KERNEL_BF16(max_pool2d, pool2d_max_bfloat16)
