#include "active_inference.cuh"

#define BELIEF_PARTIAL_BODY(loadFn) \
    __shared__ float reduction[activeThreadCountCUDA]; \
    unsigned int valueIndex = blockIdx.x * activeThreadCountCUDA + threadIdx.x; \
    float localValue = 0.0f; \
    if (valueIndex < count) { \
        localValue = loadFn(likelihood, valueIndex) * loadFn(prior, valueIndex); \
    } \
    reduction[threadIdx.x] = localValue; \
    __syncthreads(); \
    active_block_reduce(reduction, threadIdx.x); \
    if (threadIdx.x == 0u) { \
        scratch[blockIdx.x] = reduction[0]; \
    }

#define BELIEF_NORMALIZE_BODY(loadFn, storeFn) \
    __shared__ float reduction[activeThreadCountCUDA]; \
    float localSum = 0.0f; \
    for (unsigned int partialIndex = threadIdx.x; partialIndex < partialCount; partialIndex += activeThreadCountCUDA) { \
        localSum += scratch[partialIndex]; \
    } \
    reduction[threadIdx.x] = localSum; \
    __syncthreads(); \
    active_block_reduce(reduction, threadIdx.x); \
    float total = reduction[0]; \
    for (unsigned int index = threadIdx.x; index < count; index += activeThreadCountCUDA) { \
        float value = loadFn(likelihood, index) * loadFn(prior, index); \
        if (total != 0.0f) { \
            value /= total; \
        } \
        storeFn(out, index, value); \
    }

extern "C" __global__ void belief_update_float32_partial(
    const float* likelihood,
    const float* prior,
    float* scratch,
    unsigned int count
) {
    BELIEF_PARTIAL_BODY(active_load_f32)
}

extern "C" __global__ void belief_update_float32_normalize(
    const float* likelihood,
    const float* prior,
    const float* scratch,
    float* out,
    unsigned int count,
    unsigned int partialCount
) {
    BELIEF_NORMALIZE_BODY(active_load_f32, active_store_f32)
}

extern "C" __global__ void belief_update_float16_partial(
    const __half* likelihood,
    const __half* prior,
    float* scratch,
    unsigned int count
) {
    BELIEF_PARTIAL_BODY(active_load_f16)
}

extern "C" __global__ void belief_update_float16_normalize(
    const __half* likelihood,
    const __half* prior,
    const float* scratch,
    __half* out,
    unsigned int count,
    unsigned int partialCount
) {
    BELIEF_NORMALIZE_BODY(active_load_f16, active_store_f16)
}

extern "C" __global__ void belief_update_bfloat16_partial(
    const __nv_bfloat16* likelihood,
    const __nv_bfloat16* prior,
    float* scratch,
    unsigned int count
) {
    BELIEF_PARTIAL_BODY(active_load_bf16)
}

extern "C" __global__ void belief_update_bfloat16_normalize(
    const __nv_bfloat16* likelihood,
    const __nv_bfloat16* prior,
    const float* scratch,
    __nv_bfloat16* out,
    unsigned int count,
    unsigned int partialCount
) {
    BELIEF_NORMALIZE_BODY(active_load_bf16, active_store_bf16)
}

PRECISION_WEIGHT_KERNEL(
    precision_weight_float32,
    active_load_f32,
    active_store_f32,
    float
)
PRECISION_WEIGHT_KERNEL(
    precision_weight_float16,
    active_load_f16,
    active_store_f16,
    __half
)
PRECISION_WEIGHT_KERNEL(
    precision_weight_bfloat16,
    active_load_bf16,
    active_store_bf16,
    __nv_bfloat16
)
