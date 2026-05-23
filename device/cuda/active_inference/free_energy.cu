#include "active_inference.cuh"

#define FREE_ENERGY_PARTIAL_BODY(loadFn) \
    __shared__ float reduction[activeThreadCountCUDA]; \
    unsigned int valueIndex = blockIdx.x * activeThreadCountCUDA + threadIdx.x; \
    float localValue = 0.0f; \
    if (valueIndex < count) { \
        float posteriorValue = loadFn(posterior, valueIndex); \
        float safeLikelihood = active_safe_positive(loadFn(likelihood, valueIndex)); \
        float safePosterior = active_safe_positive(posteriorValue); \
        float safePrior = active_safe_positive(loadFn(prior, valueIndex)); \
        localValue = posteriorValue * (-logf(safeLikelihood) + logf(safePosterior) - logf(safePrior)); \
    } \
    reduction[threadIdx.x] = localValue; \
    __syncthreads(); \
    active_block_reduce(reduction, threadIdx.x); \
    if (threadIdx.x == 0u) { \
        scratch[blockIdx.x] = reduction[0]; \
    }

#define EXPECTED_FREE_ENERGY_PARTIAL_BODY(loadFn) \
    __shared__ float reduction[activeThreadCountCUDA]; \
    bool obsGroup = blockIdx.x < obsPartialCount; \
    unsigned int localGroup = obsGroup ? blockIdx.x : blockIdx.x - obsPartialCount; \
    unsigned int valueIndex = localGroup * activeThreadCountCUDA + threadIdx.x; \
    float localValue = 0.0f; \
    if (obsGroup && valueIndex < obsCount) { \
        float predicted = loadFn(predictedObs, valueIndex); \
        float safePredicted = active_safe_positive(predicted); \
        float safePreferred = active_safe_positive(loadFn(preferredObs, valueIndex)); \
        localValue = predicted * (logf(safePredicted) - logf(safePreferred)); \
    } \
    if (!obsGroup && valueIndex < stateCount) { \
        float state = loadFn(predictedState, valueIndex); \
        localValue = -state * logf(active_safe_positive(state)); \
    } \
    reduction[threadIdx.x] = localValue; \
    __syncthreads(); \
    active_block_reduce(reduction, threadIdx.x); \
    if (threadIdx.x == 0u) { \
        scratch[blockIdx.x] = reduction[0]; \
    }

#define ACTIVE_SCALAR_FINALIZE_BODY(storeFn) \
    __shared__ float reduction[activeThreadCountCUDA]; \
    float localValue = 0.0f; \
    for (unsigned int index = threadIdx.x; index < partialCount; index += activeThreadCountCUDA) { \
        localValue += scratch[index]; \
    } \
    reduction[threadIdx.x] = localValue; \
    __syncthreads(); \
    active_block_reduce(reduction, threadIdx.x); \
    if (threadIdx.x == 0u) { \
        storeFn(out, 0u, reduction[0]); \
    }

extern "C" __global__ void free_energy_float32_partial(
    const float* likelihood,
    const float* posterior,
    const float* prior,
    float* scratch,
    unsigned int count
) {
    FREE_ENERGY_PARTIAL_BODY(active_load_f32)
}

extern "C" __global__ void free_energy_float16_partial(
    const __half* likelihood,
    const __half* posterior,
    const __half* prior,
    float* scratch,
    unsigned int count
) {
    FREE_ENERGY_PARTIAL_BODY(active_load_f16)
}

extern "C" __global__ void free_energy_bfloat16_partial(
    const __nv_bfloat16* likelihood,
    const __nv_bfloat16* posterior,
    const __nv_bfloat16* prior,
    float* scratch,
    unsigned int count
) {
    FREE_ENERGY_PARTIAL_BODY(active_load_bf16)
}

extern "C" __global__ void expected_free_energy_float32_partial(
    const float* predictedObs,
    const float* preferredObs,
    const float* predictedState,
    float* scratch,
    unsigned int obsCount,
    unsigned int stateCount,
    unsigned int obsPartialCount
) {
    EXPECTED_FREE_ENERGY_PARTIAL_BODY(active_load_f32)
}

extern "C" __global__ void expected_free_energy_float16_partial(
    const __half* predictedObs,
    const __half* preferredObs,
    const __half* predictedState,
    float* scratch,
    unsigned int obsCount,
    unsigned int stateCount,
    unsigned int obsPartialCount
) {
    EXPECTED_FREE_ENERGY_PARTIAL_BODY(active_load_f16)
}

extern "C" __global__ void expected_free_energy_bfloat16_partial(
    const __nv_bfloat16* predictedObs,
    const __nv_bfloat16* preferredObs,
    const __nv_bfloat16* predictedState,
    float* scratch,
    unsigned int obsCount,
    unsigned int stateCount,
    unsigned int obsPartialCount
) {
    EXPECTED_FREE_ENERGY_PARTIAL_BODY(active_load_bf16)
}

extern "C" __global__ void active_scalar_finalize_float32_value(
    const float* scratch,
    float* out,
    unsigned int partialCount
) {
    ACTIVE_SCALAR_FINALIZE_BODY(active_store_f32)
}

extern "C" __global__ void active_scalar_finalize_float16_value(
    const float* scratch,
    __half* out,
    unsigned int partialCount
) {
    ACTIVE_SCALAR_FINALIZE_BODY(active_store_f16)
}

extern "C" __global__ void active_scalar_finalize_bfloat16_value(
    const float* scratch,
    __nv_bfloat16* out,
    unsigned int partialCount
) {
    ACTIVE_SCALAR_FINALIZE_BODY(active_store_bf16)
}
