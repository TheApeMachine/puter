#include "physics.metal"

using namespace metal;

#define PHYSICS_FFT_KERNELS(prefix, storage, scalar) \
kernel void prefix##_fft_bit_reverse( \
kernel void prefix##_fft_stage( \
kernel void prefix##_fft_scale( \
kernel void prefix##_dft_naive( \
PHYSICS_FFT_KERNELS(float32, Float32PhysicsStorage, float)
PHYSICS_FFT_KERNELS(float16, Float16PhysicsStorage, half)
PHYSICS_FFT_KERNELS(bfloat16, BFloat16PhysicsStorage, ushort)
