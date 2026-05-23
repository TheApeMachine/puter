#include "embedding.cuh"

static __device__ __forceinline__ float embedding_bag_load_f32(const float* table, unsigned int index) {
    return table[index];
}

static __device__ __forceinline__ void embedding_bag_store_f32(float* out, unsigned int index, float value) {
    out[index] = value;
}

static __device__ __forceinline__ float embedding_bag_load_f16(const __half* table, unsigned int index) {
    return __half2float(table[index]);
}

static __device__ __forceinline__ void embedding_bag_store_f16(__half* out, unsigned int index, float value) {
    out[index] = __float2half(value);
}

static __device__ __forceinline__ float embedding_bag_load_bf16(const __nv_bfloat16* table, unsigned int index) {
    return __bfloat162float(table[index]);
}

static __device__ __forceinline__ void embedding_bag_store_bf16(__nv_bfloat16* out, unsigned int index, float value) {
    out[index] = __float2bfloat16(value);
}

#define EMBEDDING_BAG_KERNEL(name, scalarType, loadFn, storeFn) \
extern "C" __global__ void name( \
    const scalarType* table, \
    const int* indices, \
    const int* offsets, \
    scalarType* out, \
    unsigned int* errorFlag, \
    unsigned int vocab, \
    unsigned int hidden, \
    unsigned int indexCount, \
    unsigned int bagCount \
) { \
    unsigned int outputIndex = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int total = bagCount * hidden; \
    if (outputIndex >= total) { \
        return; \
    } \
    unsigned int bagIndex = outputIndex / hidden; \
    unsigned int hiddenOffset = outputIndex - bagIndex * hidden; \
    int start = offsets[bagIndex]; \
    int end = bagIndex + 1u < bagCount ? offsets[bagIndex + 1u] : static_cast<int>(indexCount); \
    if (start < 0 || end < start || static_cast<unsigned int>(end) > indexCount) { \
        if (errorFlag != nullptr) { \
            atomicOr(errorFlag, 1u); \
        } \
        return; \
    } \
    float accumulator = 0.0f; \
    for (int indexCursor = start; indexCursor < end; indexCursor++) { \
        int tokenID = indices[indexCursor]; \
        if (tokenID < 0 || static_cast<unsigned int>(tokenID) >= vocab) { \
            if (errorFlag != nullptr) { \
                atomicOr(errorFlag, 1u); \
            } \
            return; \
        } \
        accumulator += loadFn( \
            table, static_cast<unsigned int>(tokenID) * hidden + hiddenOffset \
        ); \
    } \
    storeFn(out, outputIndex, accumulator); \
}

EMBEDDING_BAG_KERNEL(embedding_bag_float32, float, embedding_bag_load_f32, embedding_bag_store_f32)
EMBEDDING_BAG_KERNEL(embedding_bag_float16, __half, embedding_bag_load_f16, embedding_bag_store_f16)
EMBEDDING_BAG_KERNEL(embedding_bag_bfloat16, __nv_bfloat16, embedding_bag_load_bf16, embedding_bag_store_bf16)
