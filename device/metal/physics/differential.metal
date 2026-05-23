#include "physics.metal"

using namespace metal;

#define PHYSICS_LAPLACIAN_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device const scalar* spacing [[buffer(1)]], \
    device scalar* out [[buffer(2)]], \
    constant uint& count [[buffer(3)]], \
    constant uint& rank [[buffer(4)]], \
    constant uint& dim0 [[buffer(5)]], \
    constant uint& dim1 [[buffer(6)]], \
    constant uint& dim2 [[buffer(7)]], \
    uint index [[thread_position_in_grid]] \
) { \
    laplacian_kernel<storage, scalar>(input, spacing, out, count, rank, dim0, dim1, dim2, index); \
}

#define PHYSICS_VECTOR_KERNEL(name, helper, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device const scalar* spacing [[buffer(1)]], \
    device scalar* out [[buffer(2)]], \
    constant uint& count [[buffer(3)]], \
    uint index [[thread_position_in_grid]] \
) { \
    helper<storage, scalar>(input, spacing, out, count, index); \
}

#define PHYSICS_MADELUNG_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* density [[buffer(0)]], \
    device const scalar* velocity [[buffer(1)]], \
    device const scalar* spacing [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    constant uint& count [[buffer(4)]], \
    uint index [[thread_position_in_grid]] \
) { \
    madelung_continuity_kernel<storage, scalar>(density, velocity, spacing, out, count, index); \
}

    device const scalar* realIn [[buffer(0)]], \
    device const scalar* imagIn [[buffer(1)]], \
    device scalar* realOut [[buffer(2)]], \
    device scalar* imagOut [[buffer(3)]], \
    constant uint& count [[buffer(4)]], \
    constant uint& bits [[buffer(5)]], \
    uint index [[thread_position_in_grid]] \
) { \
    fft_bit_reverse_kernel<storage, scalar>(realIn, imagIn, realOut, imagOut, count, bits, index); \
} \
    device scalar* realValues [[buffer(0)]], \
    device scalar* imagValues [[buffer(1)]], \
    constant uint& length [[buffer(2)]], \
    constant uint& inverseValue [[buffer(3)]], \
    uint butterfly [[thread_position_in_grid]] \
) { \
    fft_stage_kernel<storage, scalar>(realValues, imagValues, length, inverseValue, butterfly); \
} \
    device scalar* realValues [[buffer(0)]], \
    device scalar* imagValues [[buffer(1)]], \
    constant uint& count [[buffer(2)]], \
    uint index [[thread_position_in_grid]] \
) { \
    fft_scale_kernel<storage, scalar>(realValues, imagValues, count, index); \
} \
    device const scalar* realIn [[buffer(0)]], \
    device const scalar* imagIn [[buffer(1)]], \
    device scalar* realOut [[buffer(2)]], \
    device scalar* imagOut [[buffer(3)]], \
    device const float* twiddleReal [[buffer(4)]], \
    device const float* twiddleImag [[buffer(5)]], \
    constant uint& count [[buffer(6)]], \
    constant uint& inverseValue [[buffer(7)]], \
    uint index [[thread_position_in_grid]] \
) { \
    dft_naive_kernel<storage, scalar>( \
        realIn, imagIn, realOut, imagOut, twiddleReal, twiddleImag, count, inverseValue, index \
    ); \
}

PHYSICS_LAPLACIAN_KERNEL(laplacian_float32, Float32PhysicsStorage, float)
PHYSICS_VECTOR_KERNEL(laplacian4_float32, laplacian4_kernel, Float32PhysicsStorage, float)
PHYSICS_VECTOR_KERNEL(grad1d_float32, grad1d_kernel, Float32PhysicsStorage, float)
PHYSICS_VECTOR_KERNEL(divergence1d_float32, grad1d_kernel, Float32PhysicsStorage, float)
PHYSICS_VECTOR_KERNEL(quantum_potential_float32, quantum_potential_kernel, Float32PhysicsStorage, float)
PHYSICS_VECTOR_KERNEL(bohmian_velocity_float32, bohmian_velocity_kernel, Float32PhysicsStorage, float)
PHYSICS_MADELUNG_KERNEL(madelung_continuity_float32, Float32PhysicsStorage, float)

PHYSICS_LAPLACIAN_KERNEL(laplacian_float16, Float16PhysicsStorage, half)
PHYSICS_VECTOR_KERNEL(laplacian4_float16, laplacian4_kernel, Float16PhysicsStorage, half)
PHYSICS_VECTOR_KERNEL(grad1d_float16, grad1d_kernel, Float16PhysicsStorage, half)
PHYSICS_VECTOR_KERNEL(divergence1d_float16, grad1d_kernel, Float16PhysicsStorage, half)
PHYSICS_VECTOR_KERNEL(quantum_potential_float16, quantum_potential_kernel, Float16PhysicsStorage, half)
PHYSICS_VECTOR_KERNEL(bohmian_velocity_float16, bohmian_velocity_kernel, Float16PhysicsStorage, half)
PHYSICS_MADELUNG_KERNEL(madelung_continuity_float16, Float16PhysicsStorage, half)

PHYSICS_LAPLACIAN_KERNEL(laplacian_bfloat16, BFloat16PhysicsStorage, ushort)
PHYSICS_VECTOR_KERNEL(laplacian4_bfloat16, laplacian4_kernel, BFloat16PhysicsStorage, ushort)
PHYSICS_VECTOR_KERNEL(grad1d_bfloat16, grad1d_kernel, BFloat16PhysicsStorage, ushort)
PHYSICS_VECTOR_KERNEL(divergence1d_bfloat16, grad1d_kernel, BFloat16PhysicsStorage, ushort)
PHYSICS_VECTOR_KERNEL(quantum_potential_bfloat16, quantum_potential_kernel, BFloat16PhysicsStorage, ushort)
PHYSICS_VECTOR_KERNEL(bohmian_velocity_bfloat16, bohmian_velocity_kernel, BFloat16PhysicsStorage, ushort)
PHYSICS_MADELUNG_KERNEL(madelung_continuity_bfloat16, BFloat16PhysicsStorage, ushort)
