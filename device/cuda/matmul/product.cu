#include "matmul.cuh"

static __device__ __forceinline__ float matmul_load_f32(const float* values, unsigned int index) {
    return values[index];
}

static __device__ __forceinline__ void matmul_store_f32(float* values, unsigned int index, float value) {
    values[index] = value;
}

static __device__ __forceinline__ float matmul_load_f16(const __half* values, unsigned int index) {
    return __half2float(values[index]);
}

static __device__ __forceinline__ void matmul_store_f16(__half* values, unsigned int index, float value) {
    values[index] = __float2half(value);
}

static __device__ __forceinline__ float matmul_load_bf16(const __nv_bfloat16* values, unsigned int index) {
    return __bfloat162float(values[index]);
}

static __device__ __forceinline__ void matmul_store_bf16(__nv_bfloat16* values, unsigned int index, float value) {
    values[index] = __float2bfloat16(value);
}

#define MATMUL_TILED_KERNEL(name, loadLeft, loadRight, storeOut, leftType, rightType, outType) \
extern "C" __global__ void name( \
    const leftType* left, \
    const rightType* right, \
    outType* out, \
    unsigned int rows, \
    unsigned int inner, \
    unsigned int cols \
) { \
    __shared__ float leftTile[matmulSharedFloatCountCUDA]; \
    __shared__ float rightTile[matmulSharedFloatCountCUDA]; \
    unsigned int row = blockIdx.y * matmulTileSizeCUDA + threadIdx.y; \
    unsigned int col = blockIdx.x * matmulTileSizeCUDA + threadIdx.x; \
    unsigned int localOffset = threadIdx.y * matmulTileSizeCUDA + threadIdx.x; \
    float accumulator = 0.0f; \
    for (unsigned int tileStart = 0u; tileStart < inner; tileStart += matmulTileSizeCUDA) { \
        unsigned int leftInner = tileStart + threadIdx.x; \
        unsigned int rightInner = tileStart + threadIdx.y; \
        leftTile[localOffset] = \
            row < rows && leftInner < inner ? loadLeft(left, row * inner + leftInner) : 0.0f; \
        rightTile[localOffset] = \
            rightInner < inner && col < cols ? loadRight(right, rightInner * cols + col) : 0.0f; \
        __syncthreads(); \
        for (unsigned int tileIndex = 0u; tileIndex < matmulTileSizeCUDA; tileIndex++) { \
            accumulator += leftTile[threadIdx.y * matmulTileSizeCUDA + tileIndex] * \
                rightTile[tileIndex * matmulTileSizeCUDA + threadIdx.x]; \
        } \
        __syncthreads(); \
    } \
    if (row < rows && col < cols) { \
        storeOut(out, row * cols + col, accumulator); \
    } \
}

#define MATMUL_ADD_TILED_KERNEL(name, loadLeft, loadRight, loadBias, storeOut, leftType, rightType, biasType, outType) \
extern "C" __global__ void name( \
    const leftType* left, \
    const rightType* right, \
    const biasType* bias, \
    outType* out, \
    unsigned int rows, \
    unsigned int inner, \
    unsigned int cols \
) { \
    __shared__ float leftTile[matmulSharedFloatCountCUDA]; \
    __shared__ float rightTile[matmulSharedFloatCountCUDA]; \
    unsigned int row = blockIdx.y * matmulTileSizeCUDA + threadIdx.y; \
    unsigned int col = blockIdx.x * matmulTileSizeCUDA + threadIdx.x; \
    unsigned int localOffset = threadIdx.y * matmulTileSizeCUDA + threadIdx.x; \
    float accumulator = col < cols ? loadBias(bias, col) : 0.0f; \
    for (unsigned int tileStart = 0u; tileStart < inner; tileStart += matmulTileSizeCUDA) { \
        unsigned int leftInner = tileStart + threadIdx.x; \
        unsigned int rightInner = tileStart + threadIdx.y; \
        leftTile[localOffset] = \
            row < rows && leftInner < inner ? loadLeft(left, row * inner + leftInner) : 0.0f; \
        rightTile[localOffset] = \
            rightInner < inner && col < cols ? loadRight(right, rightInner * cols + col) : 0.0f; \
        __syncthreads(); \
        for (unsigned int tileIndex = 0u; tileIndex < matmulTileSizeCUDA; tileIndex++) { \
            accumulator += leftTile[threadIdx.y * matmulTileSizeCUDA + tileIndex] * \
                rightTile[tileIndex * matmulTileSizeCUDA + threadIdx.x]; \
        } \
        __syncthreads(); \
    } \
    if (row < rows && col < cols) { \
        storeOut(out, row * cols + col, accumulator); \
    } \
}

MATMUL_TILED_KERNEL(
    matmul_float32,
    matmul_load_f32,
    matmul_load_f32,
    matmul_store_f32,
    float,
    float,
    float
)
MATMUL_TILED_KERNEL(
    matmul_float16,
    matmul_load_f16,
    matmul_load_f16,
    matmul_store_f16,
    __half,
    __half,
    __half
)
MATMUL_TILED_KERNEL(
    matmul_bfloat16,
    matmul_load_bf16,
    matmul_load_bf16,
    matmul_store_bf16,
    __nv_bfloat16,
    __nv_bfloat16,
    __nv_bfloat16
)

MATMUL_ADD_TILED_KERNEL(
    matmul_add_float32,
    matmul_load_f32,
    matmul_load_f32,
    matmul_load_f32,
    matmul_store_f32,
    float,
    float,
    float,
    float
)
MATMUL_ADD_TILED_KERNEL(
    matmul_add_float16,
    matmul_load_f16,
    matmul_load_f16,
    matmul_load_f16,
    matmul_store_f16,
    __half,
    __half,
    __half,
    __half
)
MATMUL_ADD_TILED_KERNEL(
    matmul_add_bfloat16,
    matmul_load_bf16,
    matmul_load_bf16,
    matmul_load_bf16,
    matmul_store_bf16,
    __nv_bfloat16,
    __nv_bfloat16,
    __nv_bfloat16,
    __nv_bfloat16
)
