#include "dropout.cuh"

#define DROPOUT_KERNEL(name, scalarType, loadFn, storeFn) \
extern "C" __global__ void name( \
    const scalarType* input, \
    scalarType* out, \
    unsigned int count, \
    float scale, \
    unsigned int threshold, \
    DropoutSeed seed \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (index >= count) { \
        return; \
    } \
    unsigned int randomValue = dropout_seed_for_index(index, count, seed); \
    scalarType outValue = scalarType(0); \
    if (randomValue < threshold) { \
        outValue = loadFn(input, index) * scalarType(scale); \
    } \
    storeFn(out, index, outValue); \
}

static __device__ __forceinline__ float dropout_load_f32(const float* input, unsigned int index) {
    return input[index];
}

static __device__ __forceinline__ void dropout_store_f32(float* out, unsigned int index, float value) {
    out[index] = value;
}

static __device__ __forceinline__ __half dropout_load_f16(const __half* input, unsigned int index) {
    return input[index];
}

static __device__ __forceinline__ void dropout_store_f16(__half* out, unsigned int index, __half value) {
    out[index] = value;
}

static __device__ __forceinline__ __nv_bfloat16 dropout_load_bf16(const __nv_bfloat16* input, unsigned int index) {
    return input[index];
}

static __device__ __forceinline__ void dropout_store_bf16(__nv_bfloat16* out, unsigned int index, __nv_bfloat16 value) {
    out[index] = value;
}

DROPOUT_KERNEL(dropout_float32, float, dropout_load_f32, dropout_store_f32)
DROPOUT_KERNEL(dropout_float16, __half, dropout_load_f16, dropout_store_f16)
DROPOUT_KERNEL(dropout_bfloat16, __nv_bfloat16, dropout_load_bf16, dropout_store_bf16)
