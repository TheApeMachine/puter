#ifndef PUTER_DEVICE_CUDA_PHYSICS_PHYSICS_CUH
#define PUTER_DEVICE_CUDA_PHYSICS_PHYSICS_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

static __device__ __forceinline__ float physics_load_f32(const float* values, unsigned int index) {
    return values[index];
}

static __device__ __forceinline__ void physics_store_f32(float* values, unsigned int index, float value) {
    values[index] = value;
}

static __device__ __forceinline__ float physics_load_f16(const __half* values, unsigned int index) {
    return __half2float(values[index]);
}

static __device__ __forceinline__ void physics_store_f16(__half* values, unsigned int index, float value) {
    values[index] = __float2half(value);
}

static __device__ __forceinline__ float physics_load_bf16(const __nv_bfloat16* values, unsigned int index) {
    return __bfloat162float(values[index]);
}

static __device__ __forceinline__ void physics_store_bf16(__nv_bfloat16* values, unsigned int index, float value) {
    values[index] = __float2bfloat16(value);
}

static __device__ __forceinline__ float physics_spacing_value(
    float (*loadFn)(const void*, unsigned int),
    const void* spacingPtr
) {
    float physicsDx = loadFn(spacingPtr, 0u);

    if (physicsDx <= 0.0f) {
        return 1.0f;
    }

    return physicsDx;
}

static __device__ __forceinline__ void physics_laplacian_body(
    float (*loadInput)(const void*, unsigned int),
    void (*storeOutput)(void*, unsigned int, float),
    const void* input,
    const void* spacing,
    void* out,
    unsigned int count,
    unsigned int rank,
    unsigned int dim0,
    unsigned int dim1,
    unsigned int dim2,
    unsigned int index
) {
    if (index >= count) {
        return;
    }

    float dx = physics_spacing_value(loadInput, spacing);
    float dxSquared = dx * dx;
    float center = loadInput(input, index);

    if (rank == 1u) {
        unsigned int left = (index + dim0 - 1u) % dim0;
        unsigned int right = (index + 1u) % dim0;
        float value = loadInput(input, left) - 2.0f * center + loadInput(input, right);
        storeOutput(out, index, value / dxSquared);
        return;
    }

    if (rank == 2u) {
        unsigned int row = index / dim1;
        unsigned int col = index - row * dim1;
        unsigned int up = ((row + dim0 - 1u) % dim0) * dim1 + col;
        unsigned int down = ((row + 1u) % dim0) * dim1 + col;
        unsigned int left = row * dim1 + ((col + dim1 - 1u) % dim1);
        unsigned int right = row * dim1 + ((col + 1u) % dim1);
        float value = loadInput(input, up) + loadInput(input, down) +
            loadInput(input, left) + loadInput(input, right) - 4.0f * center;
        storeOutput(out, index, value / dxSquared);
        return;
    }

    unsigned int plane = dim1 * dim2;
    unsigned int depth = index / plane;
    unsigned int rem = index - depth * plane;
    unsigned int row = rem / dim2;
    unsigned int col = rem - row * dim2;
    unsigned int dm = ((depth + dim0 - 1u) % dim0) * plane + row * dim2 + col;
    unsigned int dp = ((depth + 1u) % dim0) * plane + row * dim2 + col;
    unsigned int rm = depth * plane + ((row + dim1 - 1u) % dim1) * dim2 + col;
    unsigned int rp = depth * plane + ((row + 1u) % dim1) * dim2 + col;
    unsigned int cm = depth * plane + row * dim2 + ((col + dim2 - 1u) % dim2);
    unsigned int cp = depth * plane + row * dim2 + ((col + 1u) % dim2);
    float value = loadInput(input, dm) + loadInput(input, dp) +
        loadInput(input, rm) + loadInput(input, rp) +
        loadInput(input, cm) + loadInput(input, cp) - 6.0f * center;
    storeOutput(out, index, value / dxSquared);
}

#define PHYSICS_LAPLACIAN_TYPED(loadInput, storeOutput, scalarType, loadFn, storeFn) \
static __device__ __forceinline__ float physics_##loadInput##_typed(const void* values, unsigned int index) { \
    return loadFn((const scalarType*)values, index); \
} \
static __device__ __forceinline__ void physics_##storeOutput##_typed(void* values, unsigned int index, float value) { \
    storeFn((scalarType*)values, index, value); \
}

PHYSICS_LAPLACIAN_TYPED(load_f32, store_f32, float, physics_load_f32, physics_store_f32)
PHYSICS_LAPLACIAN_TYPED(load_f16, store_f16, __half, physics_load_f16, physics_store_f16)
PHYSICS_LAPLACIAN_TYPED(load_bf16, store_bf16, __nv_bfloat16, physics_load_bf16, physics_store_bf16)

#define PHYSICS_LAPLACIAN_KERNEL(name, scalarType, loadFn, storeFn, loadTag, storeTag) \
extern "C" __global__ void name( \
    const scalarType* input, \
    const scalarType* spacing, \
    scalarType* out, \
    unsigned int count, \
    unsigned int rank, \
    unsigned int dim0, \
    unsigned int dim1, \
    unsigned int dim2 \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    physics_laplacian_body( \
        physics_##loadTag##_typed, \
        physics_##storeTag##_typed, \
        input, \
        spacing, \
        out, \
        count, \
        rank, \
        dim0, \
        dim1, \
        dim2, \
        index \
    ); \
}

static __device__ __forceinline__ void physics_laplacian4_body(
    float (*loadInput)(const void*, unsigned int),
    void (*storeOutput)(void*, unsigned int, float),
    const void* input,
    const void* spacing,
    void* out,
    unsigned int count,
    unsigned int index
) {
    if (index >= count) {
        return;
    }

    float dx = physics_spacing_value(loadInput, spacing);
    float denominator = 12.0f * dx * dx;
    unsigned int im2 = (index + count - 2u) % count;
    unsigned int im1 = (index + count - 1u) % count;
    unsigned int ip1 = (index + 1u) % count;
    unsigned int ip2 = (index + 2u) % count;
    float value = -loadInput(input, im2) + 16.0f * loadInput(input, im1) -
        30.0f * loadInput(input, index) + 16.0f * loadInput(input, ip1) -
        loadInput(input, ip2);
    storeOutput(out, index, value / denominator);
}

#define PHYSICS_VECTOR_KERNEL(name, bodyFn, scalarType, loadTag, storeTag) \
extern "C" __global__ void name( \
    const scalarType* input, \
    const scalarType* spacing, \
    scalarType* out, \
    unsigned int count \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    bodyFn( \
        physics_##loadTag##_typed, \
        physics_##storeTag##_typed, \
        input, \
        spacing, \
        out, \
        count, \
        index \
    ); \
}

static __device__ __forceinline__ void physics_grad1d_body(
    float (*loadInput)(const void*, unsigned int),
    void (*storeOutput)(void*, unsigned int, float),
    const void* input,
    const void* spacing,
    void* out,
    unsigned int count,
    unsigned int index
) {
    if (index >= count) {
        return;
    }

    float dx = physics_spacing_value(loadInput, spacing);
    unsigned int left = (index + count - 1u) % count;
    unsigned int right = (index + 1u) % count;
    storeOutput(out, index, (loadInput(input, right) - loadInput(input, left)) / (2.0f * dx));
}

static __device__ __forceinline__ void physics_quantum_potential_body(
    float (*loadInput)(const void*, unsigned int),
    void (*storeOutput)(void*, unsigned int, float),
    const void* density,
    const void* spacing,
    void* out,
    unsigned int count,
    unsigned int index
) {
    if (index >= count) {
        return;
    }

    if (index == 0u || index + 1u == count) {
        storeOutput(out, index, 0.0f);
        return;
    }

    float rho = loadInput(density, index);

    if (rho <= 1.0e-12f) {
        storeOutput(out, index, 0.0f);
        return;
    }

    float dx = physics_spacing_value(loadInput, spacing);
    float sqrtRho = sqrtf(rho);
    float sqrtLeft = sqrtf(fmaxf(1.0e-12f, loadInput(density, index - 1u)));
    float sqrtRight = sqrtf(fmaxf(1.0e-12f, loadInput(density, index + 1u)));
    float laplacian = (sqrtRight - 2.0f * sqrtRho + sqrtLeft) / (dx * dx);
    storeOutput(out, index, -0.5f * laplacian / sqrtRho);
}

static __device__ __forceinline__ void physics_bohmian_velocity_body(
    float (*loadInput)(const void*, unsigned int),
    void (*storeOutput)(void*, unsigned int, float),
    const void* phase,
    const void* spacing,
    void* out,
    unsigned int count,
    unsigned int index
) {
    if (index >= count) {
        return;
    }

    if (index == 0u || index + 1u == count) {
        storeOutput(out, index, 0.0f);
        return;
    }

    float dx = physics_spacing_value(loadInput, spacing);
    storeOutput(
        out,
        index,
        (loadInput(phase, index + 1u) - loadInput(phase, index - 1u)) / (2.0f * dx)
    );
}

static __device__ __forceinline__ void physics_madelung_continuity_body(
    float (*loadDensity)(const void*, unsigned int),
    float (*loadVelocity)(const void*, unsigned int),
    void (*storeOutput)(void*, unsigned int, float),
    float (*loadSpacing)(const void*, unsigned int),
    const void* density,
    const void* velocity,
    const void* spacing,
    void* out,
    unsigned int count,
    unsigned int index
) {
    if (index >= count) {
        return;
    }

    if (index == 0u || index + 1u == count) {
        storeOutput(out, index, 0.0f);
        return;
    }

    float dx = PHYSICS_SPACING(loadSpacing, spacing);
    float fluxRight = loadDensity(density, index + 1u) * loadVelocity(velocity, index + 1u);
    float fluxLeft = loadDensity(density, index - 1u) * loadVelocity(velocity, index - 1u);
    storeOutput(out, index, (fluxRight - fluxLeft) / (2.0f * dx));
}

#define PHYSICS_MADELUNG_KERNEL(name, scalarType, loadTag, storeTag) \
extern "C" __global__ void name( \
    const scalarType* density, \
    const scalarType* velocity, \
    const scalarType* spacing, \
    scalarType* out, \
    unsigned int count \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    physics_madelung_continuity_body( \
        physics_##loadTag##_typed, \
        physics_##loadTag##_typed, \
        physics_##storeTag##_typed, \
        physics_##loadTag##_typed, \
        density, \
        velocity, \
        spacing, \
        out, \
        count, \
        index \
    ); \
}

static __device__ __forceinline__ void physics_fft_bit_reverse_body(
    float (*loadReal)(const void*, unsigned int),
    float (*loadImag)(const void*, unsigned int),
    void (*storeReal)(void*, unsigned int, float),
    void (*storeImag)(void*, unsigned int, float),
    const void* realIn,
    const void* imagIn,
    void* realOut,
    void* imagOut,
    unsigned int count,
    unsigned int bits,
    unsigned int index
) {
    if (index >= count) {
        return;
    }

    unsigned int reversed = __brev(index) >> (32u - bits);
    storeReal(realOut, reversed, loadReal(realIn, index));
    storeImag(imagOut, reversed, loadImag(imagIn, index));
}

static __device__ __forceinline__ void physics_fft_stage_body(
    float (*loadReal)(const void*, unsigned int),
    float (*loadImag)(const void*, unsigned int),
    void (*storeReal)(void*, unsigned int, float),
    void (*storeImag)(void*, unsigned int, float),
    void* realValues,
    void* imagValues,
    unsigned int length,
    unsigned int inverseValue,
    unsigned int butterfly
) {
    unsigned int halfLength = length >> 1u;
    unsigned int block = butterfly / halfLength;
    unsigned int offset = butterfly - block * halfLength;
    unsigned int upper = block * length + offset;
    unsigned int lower = upper + halfLength;
    float sign = inverseValue != 0u ? 1.0f : -1.0f;
    float angle = sign * 2.0f * 3.14159265358979323846f / float(length);
    float stepReal = cosf(angle);
    float stepImag = sinf(angle);
    float twiddleReal = 1.0f;
    float twiddleImag = 0.0f;

    for (unsigned int step = 0; step < offset; step++) {
        float nextReal = twiddleReal * stepReal - twiddleImag * stepImag;
        float nextImag = twiddleReal * stepImag + twiddleImag * stepReal;
        twiddleReal = nextReal;
        twiddleImag = nextImag;
    }

    float lowerReal = loadReal(realValues, lower);
    float lowerImag = loadImag(imagValues, lower);
    float tempReal = twiddleReal * lowerReal - twiddleImag * lowerImag;
    float tempImag = twiddleReal * lowerImag + twiddleImag * lowerReal;
    float upperReal = loadReal(realValues, upper);
    float upperImag = loadImag(imagValues, upper);

    storeReal(realValues, lower, upperReal - tempReal);
    storeImag(imagValues, lower, upperImag - tempImag);
    storeReal(realValues, upper, upperReal + tempReal);
    storeImag(imagValues, upper, upperImag + tempImag);
}

static __device__ __forceinline__ void physics_fft_scale_body(
    float (*loadReal)(const void*, unsigned int),
    float (*loadImag)(const void*, unsigned int),
    void (*storeReal)(void*, unsigned int, float),
    void (*storeImag)(void*, unsigned int, float),
    void* realValues,
    void* imagValues,
    unsigned int count,
    unsigned int index
) {
    if (index >= count) {
        return;
    }

    float scale = 1.0f / float(count);
    storeReal(realValues, index, loadReal(realValues, index) * scale);
    storeImag(imagValues, index, loadImag(imagValues, index) * scale);
}

static __device__ __forceinline__ void physics_dft_naive_body(
    float (*loadReal)(const void*, unsigned int),
    float (*loadImag)(const void*, unsigned int),
    void (*storeReal)(void*, unsigned int, float),
    void (*storeImag)(void*, unsigned int, float),
    const void* realIn,
    const void* imagIn,
    void* realOut,
    void* imagOut,
    const float* twiddleReal,
    const float* twiddleImag,
    unsigned int count,
    unsigned int inverseValue,
    unsigned int index
) {
    if (index >= count) {
        return;
    }

    float sumReal = 0.0f;
    float sumImag = 0.0f;

    for (unsigned int source = 0; source < count; source++) {
        unsigned int twiddleIndex = index * count + source;
        float cosine = twiddleReal[twiddleIndex];
        float sine = twiddleImag[twiddleIndex];
        float realValue = loadReal(realIn, source);
        float imagValue = loadImag(imagIn, source);
        sumReal += realValue * cosine - imagValue * sine;
        sumImag += realValue * sine + imagValue * cosine;
    }

    if (inverseValue != 0u) {
        float scale = 1.0f / float(count);
        sumReal *= scale;
        sumImag *= scale;
    }

    storeReal(realOut, index, sumReal);
    storeImag(imagOut, index, sumImag);
}

#define PHYSICS_FFT_KERNELS(prefix, scalarType, loadTag, storeTag) \
extern "C" __global__ void prefix##_fft_bit_reverse( \
    const scalarType* realIn, \
    const scalarType* imagIn, \
    scalarType* realOut, \
    scalarType* imagOut, \
    unsigned int count, \
    unsigned int bits \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    physics_fft_bit_reverse_body( \
        physics_##loadTag##_typed, \
        physics_##loadTag##_typed, \
        physics_##storeTag##_typed, \
        physics_##storeTag##_typed, \
        realIn, \
        imagIn, \
        realOut, \
        imagOut, \
        count, \
        bits, \
        index \
    ); \
} \
extern "C" __global__ void prefix##_fft_stage( \
    scalarType* realValues, \
    scalarType* imagValues, \
    unsigned int length, \
    unsigned int inverseValue \
) { \
    unsigned int butterfly = blockIdx.x * blockDim.x + threadIdx.x; \
    physics_fft_stage_body( \
        physics_##loadTag##_typed, \
        physics_##loadTag##_typed, \
        physics_##storeTag##_typed, \
        physics_##storeTag##_typed, \
        realValues, \
        imagValues, \
        length, \
        inverseValue, \
        butterfly \
    ); \
} \
extern "C" __global__ void prefix##_fft_scale( \
    scalarType* realValues, \
    scalarType* imagValues, \
    unsigned int count \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    physics_fft_scale_body( \
        physics_##loadTag##_typed, \
        physics_##loadTag##_typed, \
        physics_##storeTag##_typed, \
        physics_##storeTag##_typed, \
        realValues, \
        imagValues, \
        count, \
        index \
    ); \
} \
extern "C" __global__ void prefix##_dft_naive( \
    const scalarType* realIn, \
    const scalarType* imagIn, \
    scalarType* realOut, \
    scalarType* imagOut, \
    const float* twiddleReal, \
    const float* twiddleImag, \
    unsigned int count, \
    unsigned int inverseValue \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    physics_dft_naive_body( \
        physics_##loadTag##_typed, \
        physics_##loadTag##_typed, \
        physics_##storeTag##_typed, \
        physics_##storeTag##_typed, \
        realIn, \
        imagIn, \
        realOut, \
        imagOut, \
        twiddleReal, \
        twiddleImag, \
        count, \
        inverseValue, \
        index \
    ); \
}

#endif
