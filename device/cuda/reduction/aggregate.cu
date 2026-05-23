#include "reduction.cuh"

static __device__ __forceinline__ float reduction_load_f32(const float* values, unsigned int index) {
    return values[index];
}

static __device__ __forceinline__ void reduction_store_f32(float* values, unsigned int index, float value) {
    values[index] = value;
}

static __device__ __forceinline__ float reduction_load_f16(const __half* values, unsigned int index) {
    return __half2float(values[index]);
}

static __device__ __forceinline__ void reduction_store_f16(__half* values, unsigned int index, float value) {
    values[index] = __float2half(value);
}

static __device__ __forceinline__ float reduction_load_bf16(const __nv_bfloat16* values, unsigned int index) {
    return __bfloat162float(values[index]);
}

static __device__ __forceinline__ void reduction_store_bf16(__nv_bfloat16* values, unsigned int index, float value) {
    values[index] = __float2bfloat16(value);
}

static __device__ __forceinline__ bool reduction_is_sum_like(unsigned int operation) {
    return operation == 0u || operation == 1u || operation == 7u ||
        operation == 8u || operation == 9u || operation == 10u;
}

static __device__ __forceinline__ bool reduction_is_arg(unsigned int operation) {
    return operation == 5u || operation == 6u;
}

static __device__ __forceinline__ float reduction_identity_a(unsigned int operation) {
    if (operation == 2u) {
        return 1.0f;
    }

    if (operation == 3u || operation == 5u) {
        return 3.4028234663852886e38f;
    }

    if (operation == 4u || operation == 6u) {
        return -3.4028234663852886e38f;
    }

    return 0.0f;
}

static __device__ __forceinline__ float reduction_partial_a(float value, unsigned int operation) {
    if (operation == 7u) {
        return fabsf(value);
    }

    if (operation == 8u || operation == 9u || operation == 10u) {
        return value * value;
    }

    return value;
}

static __device__ __forceinline__ float reduction_partial_b(float value, unsigned int operation) {
    if (operation == 9u || operation == 10u) {
        return value;
    }

    return 0.0f;
}

static __device__ __forceinline__ void reduction_combine_arg(
    float* reductionA,
    float* reductionB,
    unsigned int leftIndex,
    unsigned int rightIndex,
    bool useMax
) {
    float leftValue = reductionA[leftIndex];
    float rightValue = reductionA[rightIndex];
    bool takeRight = useMax ? rightValue > leftValue : rightValue < leftValue;

    if (!takeRight) {
        return;
    }

    reductionA[leftIndex] = rightValue;
    reductionB[leftIndex] = reductionB[rightIndex];
}

static __device__ __forceinline__ void reduction_combine_pair(
    float* reductionA,
    float* reductionB,
    unsigned int operation,
    unsigned int leftIndex,
    unsigned int rightIndex
) {
    if (operation == 2u) {
        reductionA[leftIndex] *= reductionA[rightIndex];
        return;
    }

    if (operation == 3u) {
        reductionA[leftIndex] = fminf(reductionA[leftIndex], reductionA[rightIndex]);
        return;
    }

    if (operation == 4u) {
        reductionA[leftIndex] = fmaxf(reductionA[leftIndex], reductionA[rightIndex]);
        return;
    }

    if (operation == 5u || operation == 6u) {
        reduction_combine_arg(reductionA, reductionB, leftIndex, rightIndex, operation == 6u);
        return;
    }

    reductionA[leftIndex] += reductionA[rightIndex];
    reductionB[leftIndex] += reductionB[rightIndex];
}

static __device__ __forceinline__ float reduction_finalize_value(
    float accumulatedA,
    float accumulatedB,
    unsigned int operation,
    unsigned int count
) {
    if (operation == 1u) {
        return accumulatedA / static_cast<float>(count);
    }

    if (operation == 5u || operation == 6u) {
        return accumulatedB;
    }

    if (operation == 8u) {
        return sqrtf(accumulatedA);
    }

    if (operation == 9u || operation == 10u) {
        float mean = accumulatedB / static_cast<float>(count);
        float variance = accumulatedA / static_cast<float>(count) - mean * mean;
        variance = fmaxf(variance, 0.0f);

        if (operation == 10u) {
            return sqrtf(variance);
        }

        return variance;
    }

    return accumulatedA;
}

#define REDUCTION_PARTIAL_KERNEL(name, loadFn, storeFn, scalarType) \
extern "C" __global__ void name( \
    const scalarType* input, \
    float* scratchA, \
    float* scratchB, \
    unsigned int count, \
    unsigned int operation \
) { \
    __shared__ float reductionA[reductionThreadCountCUDA]; \
    __shared__ float reductionB[reductionThreadCountCUDA]; \
    unsigned int groupIndex = blockIdx.x; \
    unsigned int threadIndex = threadIdx.x; \
    unsigned int valueIndex = groupIndex * reductionThreadCountCUDA + threadIndex; \
    float localA = reduction_identity_a(operation); \
    float localB = 0.0f; \
    if (valueIndex < count) { \
        float value = loadFn(input, valueIndex); \
        localA = reduction_partial_a(value, operation); \
        localB = reduction_is_arg(operation) ? static_cast<float>(valueIndex) : reduction_partial_b(value, operation); \
    } \
    reductionA[threadIndex] = localA; \
    reductionB[threadIndex] = localB; \
    __syncthreads(); \
    for (unsigned int stride = reductionThreadCountCUDA / 2u; stride > 0u; stride >>= 1u) { \
        if (threadIndex < stride) { \
            reduction_combine_pair(reductionA, reductionB, operation, threadIndex, threadIndex + stride); \
        } \
        __syncthreads(); \
    } \
    if (threadIndex == 0u) { \
        scratchA[groupIndex] = reductionA[0]; \
        scratchB[groupIndex] = reductionB[0]; \
    } \
}

#define REDUCTION_FINALIZE_KERNEL(name, storeFn, scalarType) \
extern "C" __global__ void name( \
    const float* scratchA, \
    const float* scratchB, \
    scalarType* out, \
    unsigned int partialCount, \
    unsigned int count, \
    unsigned int operation \
) { \
    __shared__ float reductionA[reductionThreadCountCUDA]; \
    __shared__ float reductionB[reductionThreadCountCUDA]; \
    unsigned int threadIndex = threadIdx.x; \
    float localA = reduction_identity_a(operation); \
    float localB = 0.0f; \
    if (reduction_is_sum_like(operation)) { \
        localA = 0.0f; \
    } \
    for (unsigned int index = threadIndex; index < partialCount; index += reductionThreadCountCUDA) { \
        float candidateA = scratchA[index]; \
        float candidateB = scratchB[index]; \
        if (reduction_is_arg(operation)) { \
            bool takeCandidate = operation == 6u ? candidateA > localA : candidateA < localA; \
            if (takeCandidate) { \
                localA = candidateA; \
                localB = candidateB; \
            } \
            continue; \
        } \
        if (operation == 2u) { \
            localA *= candidateA; \
            continue; \
        } \
        if (operation == 3u) { \
            localA = fminf(localA, candidateA); \
            continue; \
        } \
        if (operation == 4u) { \
            localA = fmaxf(localA, candidateA); \
            continue; \
        } \
        localA += candidateA; \
        localB += candidateB; \
    } \
    reductionA[threadIndex] = localA; \
    reductionB[threadIndex] = localB; \
    __syncthreads(); \
    for (unsigned int stride = reductionThreadCountCUDA / 2u; stride > 0u; stride >>= 1u) { \
        if (threadIndex < stride) { \
            reduction_combine_pair(reductionA, reductionB, operation, threadIndex, threadIndex + stride); \
        } \
        __syncthreads(); \
    } \
    if (threadIndex == 0u) { \
        storeFn(out, 0u, reduction_finalize_value(reductionA[0], reductionB[0], operation, count)); \
    } \
}

REDUCTION_PARTIAL_KERNEL(reduction_partial_float32, reduction_load_f32, reduction_store_f32, float)
REDUCTION_PARTIAL_KERNEL(reduction_partial_float16, reduction_load_f16, reduction_store_f16, __half)
REDUCTION_PARTIAL_KERNEL(reduction_partial_bfloat16, reduction_load_bf16, reduction_store_bf16, __nv_bfloat16)

REDUCTION_FINALIZE_KERNEL(reduction_finalize_float32, reduction_store_f32, float)
REDUCTION_FINALIZE_KERNEL(reduction_finalize_float16, reduction_store_f16, __half)
REDUCTION_FINALIZE_KERNEL(reduction_finalize_bfloat16, reduction_store_bf16, __nv_bfloat16)
