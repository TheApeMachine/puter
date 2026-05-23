#ifndef PUTER_DEVICE_CUDA_ACTIVE_INFERENCE_ACTIVE_INFERENCE_CUH
#define PUTER_DEVICE_CUDA_ACTIVE_INFERENCE_ACTIVE_INFERENCE_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

static constexpr unsigned int activeThreadCountCUDA = 256u;

static __device__ __forceinline__ float active_safe_positive(float value) {
    return value > 1.0e-12f ? value : 1.0e-12f;
}

static __device__ __forceinline__ float active_load_f32(const float* values, unsigned int index) {
    return values[index];
}

static __device__ __forceinline__ void active_store_f32(float* values, unsigned int index, float value) {
    values[index] = value;
}

static __device__ __forceinline__ float active_load_f16(const __half* values, unsigned int index) {
    return __half2float(values[index]);
}

static __device__ __forceinline__ void active_store_f16(__half* values, unsigned int index, float value) {
    values[index] = __float2half(value);
}

static __device__ __forceinline__ float active_load_bf16(const __nv_bfloat16* values, unsigned int index) {
    return __bfloat162float(values[index]);
}

static __device__ __forceinline__ void active_store_bf16(__nv_bfloat16* values, unsigned int index, float value) {
    values[index] = __float2bfloat16(value);
}

static __device__ __forceinline__ void active_block_reduce(
    float* reduction,
    unsigned int threadIndex
) {
    for (unsigned int stride = activeThreadCountCUDA / 2u; stride > 0u; stride >>= 1u) {
        if (threadIndex < stride) {
            reduction[threadIndex] += reduction[threadIndex + stride];
        }

        __syncthreads();
    }
}

static __device__ __forceinline__ void free_energy_partial_f32(
    const float* likelihood,
    const float* posterior,
    const float* prior,
    float* scratch,
    unsigned int count,
    unsigned int groupIndex,
    unsigned int threadIndex
) {
    __shared__ float reduction[activeThreadCountCUDA];
    unsigned int valueIndex = groupIndex * activeThreadCountCUDA + threadIndex;
    float localValue = 0.0f;

    if (valueIndex < count) {
        float posteriorValue = active_load_f32(posterior, valueIndex);
        float safeLikelihood = active_safe_positive(active_load_f32(likelihood, valueIndex));
        float safePosterior = active_safe_positive(posteriorValue);
        float safePrior = active_safe_positive(active_load_f32(prior, valueIndex));
        localValue = posteriorValue * (-logf(safeLikelihood) + logf(safePosterior) - logf(safePrior));
    }

    reduction[threadIndex] = localValue;
    __syncthreads();
    active_block_reduce(reduction, threadIndex);

    if (threadIndex == 0u) {
        scratch[groupIndex] = reduction[0];
    }
}

static __device__ __forceinline__ void expected_free_energy_partial_f32(
    const float* predictedObs,
    const float* preferredObs,
    const float* predictedState,
    float* scratch,
    unsigned int obsCount,
    unsigned int stateCount,
    unsigned int obsPartialCount,
    unsigned int groupIndex,
    unsigned int threadIndex
) {
    __shared__ float reduction[activeThreadCountCUDA];
    bool obsGroup = groupIndex < obsPartialCount;
    unsigned int localGroup = obsGroup ? groupIndex : groupIndex - obsPartialCount;
    unsigned int valueIndex = localGroup * activeThreadCountCUDA + threadIndex;
    float localValue = 0.0f;

    if (obsGroup && valueIndex < obsCount) {
        float predicted = active_load_f32(predictedObs, valueIndex);
        float safePredicted = active_safe_positive(predicted);
        float safePreferred = active_safe_positive(active_load_f32(preferredObs, valueIndex));
        localValue = predicted * (logf(safePredicted) - logf(safePreferred));
    }

    if (!obsGroup && valueIndex < stateCount) {
        float state = active_load_f32(predictedState, valueIndex);
        localValue = -state * logf(active_safe_positive(state));
    }

    reduction[threadIndex] = localValue;
    __syncthreads();
    active_block_reduce(reduction, threadIndex);

    if (threadIndex == 0u) {
        scratch[groupIndex] = reduction[0];
    }
}

static __device__ __forceinline__ void active_scalar_finalize_f32(
    const float* scratch,
    float* out,
    unsigned int partialCount,
    unsigned int threadIndex
) {
    __shared__ float reduction[activeThreadCountCUDA];
    float localValue = 0.0f;

    for (unsigned int index = threadIndex; index < partialCount; index += activeThreadCountCUDA) {
        localValue += scratch[index];
    }

    reduction[threadIndex] = localValue;
    __syncthreads();
    active_block_reduce(reduction, threadIndex);

    if (threadIndex == 0u) {
        active_store_f32(out, 0u, reduction[0]);
    }
}

static __device__ __forceinline__ void belief_update_partial_f32(
    const float* likelihood,
    const float* prior,
    float* scratch,
    unsigned int count,
    unsigned int groupIndex,
    unsigned int threadIndex
) {
    __shared__ float reduction[activeThreadCountCUDA];
    unsigned int valueIndex = groupIndex * activeThreadCountCUDA + threadIndex;
    float localValue = 0.0f;

    if (valueIndex < count) {
        localValue = active_load_f32(likelihood, valueIndex) * active_load_f32(prior, valueIndex);
    }

    reduction[threadIndex] = localValue;
    __syncthreads();
    active_block_reduce(reduction, threadIndex);

    if (threadIndex == 0u) {
        scratch[groupIndex] = reduction[0];
    }
}

static __device__ __forceinline__ void belief_update_normalize_f32(
    const float* likelihood,
    const float* prior,
    const float* scratch,
    float* out,
    unsigned int count,
    unsigned int partialCount,
    unsigned int threadIndex
) {
    __shared__ float reduction[activeThreadCountCUDA];
    float localSum = 0.0f;

    for (unsigned int index = threadIndex; index < partialCount; index += activeThreadCountCUDA) {
        localSum += scratch[index];
    }

    reduction[threadIndex] = localSum;
    __syncthreads();
    active_block_reduce(reduction, threadIndex);

    float total = reduction[0];

    for (unsigned int index = threadIndex; index < count; index += activeThreadCountCUDA) {
        float value = active_load_f32(likelihood, index) * active_load_f32(prior, index);

        if (total != 0.0f) {
            value /= total;
        }

        active_store_f32(out, index, value);
    }
}

#define ACTIVE_FREE_ENERGY_PARTIAL_KERNEL(name, loadFn, storeFn, scalarType, partialFn) \
extern "C" __global__ void name( \
    const scalarType* likelihood, \
    const scalarType* posterior, \
    const scalarType* prior, \
    float* scratch, \
    unsigned int count \
) { \
    partialFn( \
        likelihood, \
        posterior, \
        prior, \
        scratch, \
        count, \
        blockIdx.x, \
        threadIdx.x \
    ); \
}

#define ACTIVE_EXPECTED_FREE_ENERGY_PARTIAL_KERNEL(name, loadFn, storeFn, scalarType, partialFn) \
extern "C" __global__ void name( \
    const scalarType* predictedObs, \
    const scalarType* preferredObs, \
    const scalarType* predictedState, \
    float* scratch, \
    unsigned int obsCount, \
    unsigned int stateCount, \
    unsigned int obsPartialCount \
) { \
    partialFn( \
        predictedObs, \
        preferredObs, \
        predictedState, \
        scratch, \
        obsCount, \
        stateCount, \
        obsPartialCount, \
        blockIdx.x, \
        threadIdx.x \
    ); \
}

#define ACTIVE_SCALAR_FINALIZE_KERNEL(name, storeFn, scalarType, finalizeFn) \
extern "C" __global__ void name( \
    const float* scratch, \
    scalarType* out, \
    unsigned int partialCount \
) { \
    finalizeFn(scratch, out, partialCount, threadIdx.x); \
}

#define BELIEF_PARTIAL_KERNEL(name, scalarType, partialFn) \
extern "C" __global__ void name( \
    const scalarType* likelihood, \
    const scalarType* prior, \
    float* scratch, \
    unsigned int count \
) { \
    partialFn(likelihood, prior, scratch, count, blockIdx.x, threadIdx.x); \
}

#define BELIEF_NORMALIZE_KERNEL(name, scalarType, normalizeFn) \
extern "C" __global__ void name( \
    const scalarType* likelihood, \
    const scalarType* prior, \
    const float* scratch, \
    scalarType* out, \
    unsigned int count, \
    unsigned int partialCount \
) { \
    normalizeFn(likelihood, prior, scratch, out, count, partialCount, threadIdx.x); \
}

#define PRECISION_WEIGHT_KERNEL(name, loadFn, storeFn, scalarType) \
extern "C" __global__ void name( \
    const scalarType* errors, \
    const scalarType* precision, \
    scalarType* out, \
    unsigned int count \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (index >= count) { \
        return; \
    } \
    storeFn(out, index, loadFn(errors, index) * loadFn(precision, index)); \
}

#endif
