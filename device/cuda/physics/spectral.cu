#include "physics.cuh"

PHYSICS_FFT_KERNELS(float32, float, load_f32, store_f32)
PHYSICS_FFT_KERNELS(float16, __half, load_f16, store_f16)
PHYSICS_FFT_KERNELS(bfloat16, __nv_bfloat16, load_bf16, store_bf16)
