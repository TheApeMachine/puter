#include "dot.cuh"

__device__ __forceinline__ double dot_accumulate_f32(float left, float right) {
    return static_cast<double>(left) * static_cast<double>(right);
}

__device__ __forceinline__ float dot_load_f32(const float* values, unsigned int index) {
    return values[index];
}

__device__ __forceinline__ float dot_load_f16(const __half* values, unsigned int index) {
    return __half2float(values[index]);
}

__device__ __forceinline__ float dot_load_bf16(const __nv_bfloat16* values, unsigned int index) {
    return __bfloat162float(values[index]);
}

__device__ __forceinline__ void dot_store_f32(float* values, unsigned int index, float value) {
    values[index] = value;
}

__device__ __forceinline__ void dot_store_f16(__half* values, unsigned int index, float value) {
    values[index] = __float2half(value);
}

__device__ __forceinline__ void dot_store_bf16(__nv_bfloat16* values, unsigned int index, float value) {
    values[index] = __float2bfloat16(value);
}

#define DOT_PARTIAL_KERNEL(name, loadFn, scalarType) \
extern "C" __global__ void name##_partial( \
    const scalarType* left, \
    const scalarType* right, \
    double* scratch, \
    unsigned int count \
) { \
    __shared__ double reduction[dotThreadCountCUDA]; \
    unsigned int groupIndex = blockIdx.x; \
    unsigned int threadIndex = threadIdx.x; \
    unsigned int valueIndex = groupIndex * dotThreadCountCUDA + threadIndex; \
    double localSum = 0.0; \
    if (valueIndex < count) { \
        localSum += dot_accumulate_f32(loadFn(left, valueIndex), loadFn(right, valueIndex)); \
    } \
    reduction[threadIndex] = localSum; \
    __syncthreads(); \
    for (unsigned int stride = dotThreadCountCUDA / 2u; stride > 0u; stride >>= 1u) { \
        if (threadIndex < stride) { \
            reduction[threadIndex] += reduction[threadIndex + stride]; \
        } \
        __syncthreads(); \
    } \
    if (threadIndex == 0u) { \
        scratch[groupIndex] = reduction[0]; \
    } \
}

#define DOT_FINALIZE_KERNEL(name, storeFn, scalarType) \
extern "C" __global__ void name##_finalize( \
    const double* scratch, \
    scalarType* out, \
    unsigned int partialCount \
) { \
    __shared__ double reduction[dotThreadCountCUDA]; \
    double localSum = 0.0; \
    for (unsigned int index = threadIdx.x; index < partialCount; index += dotThreadCountCUDA) { \
        localSum += scratch[index]; \
    } \
    reduction[threadIdx.x] = localSum; \
    __syncthreads(); \
    for (unsigned int stride = dotThreadCountCUDA / 2u; stride > 0u; stride >>= 1u) { \
        if (threadIdx.x < stride) { \
            reduction[threadIdx.x] += reduction[threadIdx.x + stride]; \
        } \
        __syncthreads(); \
    } \
    if (threadIdx.x == 0u) { \
        storeFn(out, 0u, static_cast<float>(reduction[0])); \
    } \
}

DOT_PARTIAL_KERNEL(dot_float32, dot_load_f32, float)
DOT_PARTIAL_KERNEL(dot_float16, dot_load_f16, __half)
DOT_PARTIAL_KERNEL(dot_bfloat16, dot_load_bf16, __nv_bfloat16)

DOT_FINALIZE_KERNEL(dot_float32, dot_store_f32, float)
DOT_FINALIZE_KERNEL(dot_float16, dot_store_f16, __half)
DOT_FINALIZE_KERNEL(dot_bfloat16, dot_store_bf16, __nv_bfloat16)
