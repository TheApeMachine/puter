#include "pool.metal"

using namespace metal;

    adaptive_pool2d_kernel<storage, scalar>( \
ADAPTIVE_POOL2D_KERNEL(adaptive_avg_pool2d_float32, Float32VisionStorage, float)
ADAPTIVE_POOL2D_KERNEL(adaptive_avg_pool2d_float16, Float16VisionStorage, half)
ADAPTIVE_POOL2D_KERNEL(adaptive_avg_pool2d_bfloat16, BFloat16VisionStorage, ushort)
ADAPTIVE_POOL2D_KERNEL(adaptive_max_pool2d_float32, Float32VisionStorage, float)
ADAPTIVE_POOL2D_KERNEL(adaptive_max_pool2d_float16, Float16VisionStorage, half)
ADAPTIVE_POOL2D_KERNEL(adaptive_max_pool2d_bfloat16, BFloat16VisionStorage, ushort)
