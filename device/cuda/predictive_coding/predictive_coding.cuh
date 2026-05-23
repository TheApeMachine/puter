#ifndef PUTER_DEVICE_CUDA_PREDICTIVE_CODING_PREDICTIVE_CODING_CUH
#define PUTER_DEVICE_CUDA_PREDICTIVE_CODING_PREDICTIVE_CODING_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

static __device__ __forceinline__ float pc_load_f32(const float* values, unsigned int index) {
    return values[index];
}

static __device__ __forceinline__ void pc_store_f32(float* values, unsigned int index, float value) {
    values[index] = value;
}

static __device__ __forceinline__ float pc_load_f16(const __half* values, unsigned int index) {
    return __half2float(values[index]);
}

static __device__ __forceinline__ void pc_store_f16(__half* values, unsigned int index, float value) {
    values[index] = __float2half(value);
}

static __device__ __forceinline__ float pc_load_bf16(const __nv_bfloat16* values, unsigned int index) {
    return __bfloat162float(values[index]);
}

static __device__ __forceinline__ void pc_store_bf16(__nv_bfloat16* values, unsigned int index, float value) {
    values[index] = __float2bfloat16(value);
}

#define PC_PREDICTION_KERNEL(name, loadFn, storeFn, scalarType) \
extern "C" __global__ void name( \
    const scalarType* weights, \
    const scalarType* state, \
    scalarType* out, \
    unsigned int inCount \
) { \
    unsigned int outIndex = blockIdx.x * blockDim.x + threadIdx.x; \
    float sum = 0.0f; \
    unsigned int rowOffset = outIndex * inCount; \
    for (unsigned int inIndex = 0u; inIndex < inCount; inIndex++) { \
        sum += loadFn(weights, rowOffset + inIndex) * loadFn(state, inIndex); \
    } \
    storeFn(out, outIndex, sum); \
}

#define PC_PREDICTION_ERROR_KERNEL(name, loadFn, storeFn, scalarType) \
extern "C" __global__ void name( \
    const scalarType* observed, \
    const scalarType* predicted, \
    scalarType* out, \
    unsigned int count \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (index >= count) { \
        return; \
    } \
    storeFn(out, index, loadFn(observed, index) - loadFn(predicted, index)); \
}

#define PC_UPDATE_REPRESENTATION_KERNEL(name, loadFn, storeFn, scalarType) \
extern "C" __global__ void name( \
    const scalarType* weights, \
    const scalarType* state, \
    const scalarType* predictionError, \
    scalarType* out, \
    unsigned int outCount, \
    unsigned int inCount, \
    float learningRate \
) { \
    unsigned int inIndex = blockIdx.x * blockDim.x + threadIdx.x; \
    if (inIndex >= inCount) { \
        return; \
    } \
    float value = loadFn(state, inIndex); \
    for (unsigned int outIndex = 0u; outIndex < outCount; outIndex++) { \
        value += learningRate * \
            loadFn(weights, outIndex * inCount + inIndex) * \
            loadFn(predictionError, outIndex); \
    } \
    storeFn(out, inIndex, value); \
}

#define PC_UPDATE_WEIGHTS_KERNEL(name, loadFn, storeFn, scalarType) \
extern "C" __global__ void name( \
    const scalarType* weights, \
    const scalarType* state, \
    const scalarType* predictionError, \
    scalarType* out, \
    unsigned int inCount, \
    unsigned int count, \
    float learningRate \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (index >= count) { \
        return; \
    } \
    unsigned int outIndex = index / inCount; \
    unsigned int inIndex = index - outIndex * inCount; \
    float value = loadFn(weights, index) + \
        learningRate * \
        loadFn(predictionError, outIndex) * \
        loadFn(state, inIndex); \
    storeFn(out, index, value); \
}

#endif
