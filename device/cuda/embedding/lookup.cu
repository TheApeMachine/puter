#include "embedding.cuh"

#define EMBEDDING_LOOKUP_KERNEL(name, loadFn, storeFn, scalarType) \
extern "C" __global__ void name( \
    const scalarType* table, \
    const int* indices, \
    scalarType* out, \
    unsigned int* errorFlag, \
    unsigned int vocab, \
    unsigned int hidden, \
    unsigned int indexCount \
) { \
    unsigned int outputIndex = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int total = indexCount * hidden; \
    if (outputIndex >= total) { \
        return; \
    } \
    unsigned int tokenOffset = outputIndex / hidden; \
    unsigned int hiddenOffset = outputIndex - tokenOffset * hidden; \
    int tokenID = indices[tokenOffset]; \
    if (tokenID < 0 || static_cast<unsigned int>(tokenID) >= vocab) { \
        if (errorFlag != nullptr) { \
            atomicOr(errorFlag, 1u); \
        } \
        return; \
    } \
    out[outputIndex] = loadFn(table, static_cast<unsigned int>(tokenID) * hidden + hiddenOffset); \
}

static __device__ __forceinline__ float embedding_load_f32(const float* table, unsigned int index) {
    return table[index];
}

static __device__ __forceinline__ void embedding_store_f32(float* out, unsigned int index, float value) {
    out[index] = value;
}

static __device__ __forceinline__ __half embedding_load_f16(const __half* table, unsigned int index) {
    return table[index];
}

static __device__ __forceinline__ void embedding_store_f16(__half* out, unsigned int index, __half value) {
    out[index] = value;
}

static __device__ __forceinline__ __nv_bfloat16 embedding_load_bf16(const __nv_bfloat16* table, unsigned int index) {
    return table[index];
}

static __device__ __forceinline__ void embedding_store_bf16(__nv_bfloat16* out, unsigned int index, __nv_bfloat16 value) {
    out[index] = value;
}

EMBEDDING_LOOKUP_KERNEL(embedding_lookup_float32, embedding_load_f32, embedding_store_f32, float)
EMBEDDING_LOOKUP_KERNEL(embedding_lookup_float16, embedding_load_f16, embedding_store_f16, __half)
EMBEDDING_LOOKUP_KERNEL(embedding_lookup_bfloat16, embedding_load_bf16, embedding_store_bf16, __nv_bfloat16)
