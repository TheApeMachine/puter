#ifndef PUTER_DEVICE_CUDA_CAUSAL_CAUSAL_CUH
#define PUTER_DEVICE_CUDA_CAUSAL_CAUSAL_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

static constexpr unsigned int causalThreadCountCUDA = 256u;

static __device__ __forceinline__ float causal_load_f32(const float* values, unsigned int index) {
    return values[index];
}

static __device__ __forceinline__ void causal_store_f32(float* values, unsigned int index, float value) {
    values[index] = value;
}

static __device__ __forceinline__ float causal_load_f16(const __half* values, unsigned int index) {
    return __half2float(values[index]);
}

static __device__ __forceinline__ void causal_store_f16(__half* values, unsigned int index, float value) {
    values[index] = __float2half(value);
}

static __device__ __forceinline__ float causal_load_bf16(const __nv_bfloat16* values, unsigned int index) {
    return __bfloat162float(values[index]);
}

static __device__ __forceinline__ void causal_store_bf16(__nv_bfloat16* values, unsigned int index, float value) {
    values[index] = __float2bfloat16(value);
}

static __device__ __forceinline__ float causal_safe_positive(float value) {
    return fmaxf(value, 1.0e-12f);
}

static __device__ __forceinline__ void causal_reduce_sum(float* values, unsigned int threadIndex) {
    __syncthreads();

    for (unsigned int stride = causalThreadCountCUDA / 2u; stride > 0u; stride >>= 1u) {
        if (threadIndex < stride) {
            values[threadIndex] += values[threadIndex + stride];
        }

        __syncthreads();
    }
}

static __device__ __forceinline__ void causal_reduce_product(float* values, unsigned int threadIndex) {
    __syncthreads();

    for (unsigned int stride = causalThreadCountCUDA / 2u; stride > 0u; stride >>= 1u) {
        if (threadIndex < stride) {
            values[threadIndex] *= values[threadIndex + stride];
        }

        __syncthreads();
    }
}

#define CAUSAL_BACKDOOR_KERNEL(name, scalarType, loadFn, storeFn) \
extern "C" __global__ void name( \
    const scalarType* conditional, \
    const scalarType* marginal, \
    scalarType* out, \
    unsigned int xCount, \
    unsigned int zCount, \
    unsigned int yCount \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int outputCount = xCount * yCount; \
    if (index >= outputCount) { \
        return; \
    } \
    unsigned int xIndex = index / yCount; \
    unsigned int yIndex = index % yCount; \
    float total = 0.0f; \
    for (unsigned int zIndex = 0u; zIndex < zCount; zIndex++) { \
        unsigned int conditionalIndex = (xIndex * zCount + zIndex) * yCount + yIndex; \
        total += loadFn(conditional, conditionalIndex) * loadFn(marginal, zIndex); \
    } \
    storeFn(out, index, total); \
}

#define CAUSAL_FRONTDOOR_KERNEL(name, scalarType, loadFn, storeFn) \
extern "C" __global__ void name( \
    const scalarType* mediator, \
    const scalarType* outcome, \
    const scalarType* marginal, \
    scalarType* out, \
    unsigned int xCount, \
    unsigned int mCount, \
    unsigned int yCount \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int outputCount = xCount * yCount; \
    if (index >= outputCount) { \
        return; \
    } \
    unsigned int xIndex = index / yCount; \
    unsigned int yIndex = index % yCount; \
    float total = 0.0f; \
    for (unsigned int mIndex = 0u; mIndex < mCount; mIndex++) { \
        float mediatorValue = loadFn(mediator, xIndex * mCount + mIndex); \
        float innerSum = 0.0f; \
        for (unsigned int xPrimeIndex = 0u; xPrimeIndex < xCount; xPrimeIndex++) { \
            unsigned int outcomeIndex = (xPrimeIndex * mCount + mIndex) * yCount + yIndex; \
            innerSum += loadFn(outcome, outcomeIndex) * loadFn(marginal, xPrimeIndex); \
        } \
        total += mediatorValue * innerSum; \
    } \
    storeFn(out, index, total); \
}

#define CAUSAL_DO_INTERVENE_KERNEL(name, scalarType, loadFn, storeFn) \
extern "C" __global__ void name( \
    const scalarType* adjacency, \
    const int* intervened, \
    scalarType* out, \
    unsigned int nodeCount, \
    unsigned int intervenedCount \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int matrixCount = nodeCount * nodeCount; \
    if (index >= matrixCount) { \
        return; \
    } \
    unsigned int targetNode = index % nodeCount; \
    bool removeIncoming = false; \
    for (unsigned int nodeIndex = 0u; nodeIndex < intervenedCount; nodeIndex++) { \
        int candidate = intervened[nodeIndex]; \
        if (candidate >= 0 && static_cast<unsigned int>(candidate) == targetNode) { \
            removeIncoming = true; \
        } \
    } \
    storeFn(out, index, removeIncoming ? 0.0f : loadFn(adjacency, index)); \
}

#define CAUSAL_CATE_KERNEL(name, scalarType, loadFn, storeFn) \
extern "C" __global__ void name( \
    const scalarType* treated, \
    const scalarType* control, \
    scalarType* out, \
    unsigned int count \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (index < count) { \
        storeFn(out, index, loadFn(treated, index) - loadFn(control, index)); \
    } \
}

#define CAUSAL_COUNTERFACTUAL_KERNEL(name, scalarType, loadFn, storeFn) \
extern "C" __global__ void name( \
    const scalarType* observedY, \
    const scalarType* observedX, \
    const scalarType* counterfactualX, \
    const scalarType* slope, \
    scalarType* out, \
    unsigned int count \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (index >= count) { \
        return; \
    } \
    float slopeValue = loadFn(slope, 0u); \
    float value = loadFn(observedY, index) + \
        slopeValue * (loadFn(counterfactualX, index) - loadFn(observedX, index)); \
    storeFn(out, index, value); \
}

#define CAUSAL_IV_PARTIAL_KERNEL(name, scalarType, loadFn) \
extern "C" __global__ void name( \
    const scalarType* instrument, \
    const scalarType* treatment, \
    const scalarType* outcome, \
    float* scratch, \
    unsigned int count \
) { \
    __shared__ float sumZ[causalThreadCountCUDA]; \
    __shared__ float sumX[causalThreadCountCUDA]; \
    __shared__ float sumY[causalThreadCountCUDA]; \
    __shared__ float sumZY[causalThreadCountCUDA]; \
    __shared__ float sumZX[causalThreadCountCUDA]; \
    unsigned int groupIndex = blockIdx.x; \
    unsigned int threadIndex = threadIdx.x; \
    unsigned int valueIndex = groupIndex * causalThreadCountCUDA + threadIndex; \
    float zValue = 0.0f; \
    float xValue = 0.0f; \
    float yValue = 0.0f; \
    if (valueIndex < count) { \
        zValue = loadFn(instrument, valueIndex); \
        xValue = loadFn(treatment, valueIndex); \
        yValue = loadFn(outcome, valueIndex); \
    } \
    sumZ[threadIndex] = zValue; \
    sumX[threadIndex] = xValue; \
    sumY[threadIndex] = yValue; \
    sumZY[threadIndex] = zValue * yValue; \
    sumZX[threadIndex] = zValue * xValue; \
    causal_reduce_sum(sumZ, threadIndex); \
    causal_reduce_sum(sumX, threadIndex); \
    causal_reduce_sum(sumY, threadIndex); \
    causal_reduce_sum(sumZY, threadIndex); \
    causal_reduce_sum(sumZX, threadIndex); \
    if (threadIndex == 0u) { \
        unsigned int offset = groupIndex * 5u; \
        scratch[offset] = sumZ[0]; \
        scratch[offset + 1u] = sumX[0]; \
        scratch[offset + 2u] = sumY[0]; \
        scratch[offset + 3u] = sumZY[0]; \
        scratch[offset + 4u] = sumZX[0]; \
    } \
}

#define CAUSAL_IV_FINALIZE_KERNEL(name, scalarType, storeFn) \
extern "C" __global__ void name( \
    const float* scratch, \
    scalarType* out, \
    unsigned int count, \
    unsigned int partialCount \
) { \
    __shared__ float sumZ[causalThreadCountCUDA]; \
    __shared__ float sumX[causalThreadCountCUDA]; \
    __shared__ float sumY[causalThreadCountCUDA]; \
    __shared__ float sumZY[causalThreadCountCUDA]; \
    __shared__ float sumZX[causalThreadCountCUDA]; \
    unsigned int threadIndex = threadIdx.x; \
    float localZ = 0.0f; \
    float localX = 0.0f; \
    float localY = 0.0f; \
    float localZY = 0.0f; \
    float localZX = 0.0f; \
    for (unsigned int partialIndex = threadIndex; partialIndex < partialCount; partialIndex += causalThreadCountCUDA) { \
        unsigned int offset = partialIndex * 5u; \
        localZ += scratch[offset]; \
        localX += scratch[offset + 1u]; \
        localY += scratch[offset + 2u]; \
        localZY += scratch[offset + 3u]; \
        localZX += scratch[offset + 4u]; \
    } \
    sumZ[threadIndex] = localZ; \
    sumX[threadIndex] = localX; \
    sumY[threadIndex] = localY; \
    sumZY[threadIndex] = localZY; \
    sumZX[threadIndex] = localZX; \
    causal_reduce_sum(sumZ, threadIndex); \
    causal_reduce_sum(sumX, threadIndex); \
    causal_reduce_sum(sumY, threadIndex); \
    causal_reduce_sum(sumZY, threadIndex); \
    causal_reduce_sum(sumZX, threadIndex); \
    if (threadIndex == 0u) { \
        float denominator = sumZX[0] - (sumZ[0] * sumX[0]) / static_cast<float>(count); \
        float numerator = sumZY[0] - (sumZ[0] * sumY[0]) / static_cast<float>(count); \
        storeFn(out, 0u, fabsf(denominator) < 1.0e-12f ? 0.0f : numerator / denominator); \
    } \
}

#define CAUSAL_DAG_PARTIAL_KERNEL(name, scalarType, loadFn) \
extern "C" __global__ void name( \
    const scalarType* conditionals, \
    float* scratch, \
    unsigned int count \
) { \
    __shared__ float products[causalThreadCountCUDA]; \
    unsigned int groupIndex = blockIdx.x; \
    unsigned int threadIndex = threadIdx.x; \
    unsigned int valueIndex = groupIndex * causalThreadCountCUDA + threadIndex; \
    products[threadIndex] = valueIndex < count ? \
        causal_safe_positive(loadFn(conditionals, valueIndex)) : 1.0f; \
    causal_reduce_product(products, threadIndex); \
    if (threadIndex == 0u) { \
        scratch[groupIndex] = products[0]; \
    } \
}

#define CAUSAL_DAG_FINALIZE_KERNEL(name, scalarType, storeFn) \
extern "C" __global__ void name( \
    const float* scratch, \
    scalarType* out, \
    unsigned int partialCount \
) { \
    __shared__ float products[causalThreadCountCUDA]; \
    unsigned int threadIndex = threadIdx.x; \
    float localProduct = 1.0f; \
    for (unsigned int partialIndex = threadIndex; partialIndex < partialCount; partialIndex += causalThreadCountCUDA) { \
        localProduct *= scratch[partialIndex]; \
    } \
    products[threadIndex] = localProduct; \
    causal_reduce_product(products, threadIndex); \
    if (threadIndex == 0u) { \
        storeFn(out, 0u, products[0]); \
    } \
}

#endif
