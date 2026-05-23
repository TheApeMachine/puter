#include "physics.metal"

using namespace metal;

PHYSICS_FFT_KERNELS(float32, Float32PhysicsStorage, float)
PHYSICS_FFT_KERNELS(float16, Float16PhysicsStorage, half)
PHYSICS_FFT_KERNELS(bfloat16, BFloat16PhysicsStorage, ushort)
