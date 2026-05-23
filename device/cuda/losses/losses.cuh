#ifndef PUTER_DEVICE_CUDA_LOSSES_LOSSES_CUH
#define PUTER_DEVICE_CUDA_LOSSES_LOSSES_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

static constexpr unsigned int lossThreadCountCUDA = 256u;

static __device__ __forceinline__ float loss_load_f32(const float* values, unsigned int index) {
    return values[index];
}

static __device__ __forceinline__ void loss_store_f32(float* values, unsigned int index, float value) {
    values[index] = value;
}

static __device__ __forceinline__ float loss_load_f16(const __half* values, unsigned int index) {
    return __half2float(values[index]);
}

static __device__ __forceinline__ void loss_store_f16(__half* values, unsigned int index, float value) {
    values[index] = __float2half(value);
}

static __device__ __forceinline__ float loss_load_bf16(const __nv_bfloat16* values, unsigned int index) {
    return __bfloat162float(values[index]);
}

static __device__ __forceinline__ void loss_store_bf16(__nv_bfloat16* values, unsigned int index, float value) {
    values[index] = __float2bfloat16(value);
}

static __device__ __forceinline__ float loss_safe_probability(float value) {
    return fminf(fmaxf(value, 1.0e-7f), 1.0f - 1.0e-7f);
}

static __device__ __forceinline__ float loss_safe_positive(float value) {
    return fmaxf(value, 1.0e-12f);
}

static __device__ __forceinline__ float loss_mse(float prediction, float target) {
    float delta = prediction - target;
    return delta * delta;
}

static __device__ __forceinline__ float loss_mae(float prediction, float target) {
    return fabsf(prediction - target);
}

static __device__ __forceinline__ float loss_huber(float prediction, float target) {
    float delta = prediction - target;
    float magnitude = fabsf(delta);

    if (magnitude <= 1.0f) {
        return 0.5f * delta * delta;
    }

    return magnitude - 0.5f;
}

static __device__ __forceinline__ float loss_binary_cross_entropy(float prediction, float target) {
    float safePrediction = loss_safe_probability(prediction);
    return -target * logf(safePrediction) -
        (1.0f - target) * logf(1.0f - safePrediction);
}

static __device__ __forceinline__ float loss_kl_divergence(float prediction, float target) {
    float safePrediction = loss_safe_positive(prediction);
    float safeTarget = loss_safe_positive(target);
    return safeTarget * logf(safeTarget / safePrediction);
}

#define PAIR_LOSS_PARTIAL_KERNEL(name, scalarType, loadFn, storeFn, lossFn) \
extern "C" __global__ void name##_partial( \
    const scalarType* predictions, \
    const scalarType* targets, \
    float* scratch, \
    unsigned int count \
) { \
    __shared__ float reduction[lossThreadCountCUDA]; \
    unsigned int groupIndex = blockIdx.x; \
    unsigned int threadIndex = threadIdx.x; \
    unsigned int valueIndex = groupIndex * lossThreadCountCUDA + threadIndex; \
    float localValue = 0.0f; \
    if (valueIndex < count) { \
        localValue = lossFn( \
            loadFn(predictions, valueIndex), \
            loadFn(targets, valueIndex) \
        ); \
    } \
    reduction[threadIndex] = localValue; \
    __syncthreads(); \
    for (unsigned int stride = lossThreadCountCUDA / 2u; stride > 0u; stride >>= 1u) { \
        if (threadIndex < stride) { \
            reduction[threadIndex] += reduction[threadIndex + stride]; \
        } \
        __syncthreads(); \
    } \
    if (threadIndex == 0u) { \
        scratch[groupIndex] = reduction[0]; \
    } \
}

#define LOSS_FINALIZE_KERNEL(name, scalarType, storeFn) \
extern "C" __global__ void name( \
    const float* scratch, \
    scalarType* out, \
    unsigned int partialCount, \
    unsigned int denominator \
) { \
    __shared__ float reduction[lossThreadCountCUDA]; \
    float localValue = 0.0f; \
    for (unsigned int index = threadIdx.x; index < partialCount; index += lossThreadCountCUDA) { \
        localValue += scratch[index]; \
    } \
    reduction[threadIdx.x] = localValue; \
    __syncthreads(); \
    for (unsigned int stride = lossThreadCountCUDA / 2u; stride > 0u; stride >>= 1u) { \
        if (threadIdx.x < stride) { \
            reduction[threadIdx.x] += reduction[threadIdx.x + stride]; \
        } \
        __syncthreads(); \
    } \
    if (threadIdx.x == 0u) { \
        storeFn(out, 0u, reduction[0] / static_cast<float>(denominator)); \
    } \
}

PAIR_LOSS_PARTIAL_KERNEL(mse_loss_float32, float, loss_load_f32, loss_store_f32, loss_mse)
PAIR_LOSS_PARTIAL_KERNEL(mae_loss_float32, float, loss_load_f32, loss_store_f32, loss_mae)
PAIR_LOSS_PARTIAL_KERNEL(huber_loss_float32, float, loss_load_f32, loss_store_f32, loss_huber)
PAIR_LOSS_PARTIAL_KERNEL(binary_cross_entropy_float32, float, loss_load_f32, loss_store_f32, loss_binary_cross_entropy)
PAIR_LOSS_PARTIAL_KERNEL(kl_divergence_float32, float, loss_load_f32, loss_store_f32, loss_kl_divergence)
LOSS_FINALIZE_KERNEL(loss_finalize_float32, float, loss_store_f32)

PAIR_LOSS_PARTIAL_KERNEL(mse_loss_float16, __half, loss_load_f16, loss_store_f16, loss_mse)
PAIR_LOSS_PARTIAL_KERNEL(mae_loss_float16, __half, loss_load_f16, loss_store_f16, loss_mae)
PAIR_LOSS_PARTIAL_KERNEL(huber_loss_float16, __half, loss_load_f16, loss_store_f16, loss_huber)
PAIR_LOSS_PARTIAL_KERNEL(binary_cross_entropy_float16, __half, loss_load_f16, loss_store_f16, loss_binary_cross_entropy)
PAIR_LOSS_PARTIAL_KERNEL(kl_divergence_float16, __half, loss_load_f16, loss_store_f16, loss_kl_divergence)
LOSS_FINALIZE_KERNEL(loss_finalize_float16, __half, loss_store_f16)

PAIR_LOSS_PARTIAL_KERNEL(mse_loss_bfloat16, __nv_bfloat16, loss_load_bf16, loss_store_bf16, loss_mse)
PAIR_LOSS_PARTIAL_KERNEL(mae_loss_bfloat16, __nv_bfloat16, loss_load_bf16, loss_store_bf16, loss_mae)
PAIR_LOSS_PARTIAL_KERNEL(huber_loss_bfloat16, __nv_bfloat16, loss_load_bf16, loss_store_bf16, loss_huber)
PAIR_LOSS_PARTIAL_KERNEL(binary_cross_entropy_bfloat16, __nv_bfloat16, loss_load_bf16, loss_store_bf16, loss_binary_cross_entropy)
PAIR_LOSS_PARTIAL_KERNEL(kl_divergence_bfloat16, __nv_bfloat16, loss_load_bf16, loss_store_bf16, loss_kl_divergence)
LOSS_FINALIZE_KERNEL(loss_finalize_bfloat16, __nv_bfloat16, loss_store_bf16)

#define CROSS_ENTROPY_PARTIAL_KERNEL(name, scalarType, loadFn) \
extern "C" __global__ void name##_partial( \
    const scalarType* logits, \
    const int* targets, \
    float* scratch, \
    unsigned int* errorFlag, \
    unsigned int batch, \
    unsigned int classes \
) { \
    unsigned int rowIndex = blockIdx.x; \
    if (rowIndex >= batch) { \
        return; \
    } \
    __shared__ float reduction[lossThreadCountCUDA]; \
    unsigned int threadIndex = threadIdx.x; \
    int targetID = targets[rowIndex]; \
    bool targetOK = targetID >= 0 && static_cast<unsigned int>(targetID) < classes; \
    unsigned int rowOffset = rowIndex * classes; \
    float localMax = -CUDART_INF_F; \
    if (!targetOK && threadIndex == 0u && errorFlag != nullptr) { \
        atomicOr(errorFlag, 1u); \
    } \
    for (unsigned int col = threadIndex; targetOK && col < classes; col += lossThreadCountCUDA) { \
        localMax = fmaxf(localMax, loadFn(logits, rowOffset + col)); \
    } \
    reduction[threadIndex] = localMax; \
    __syncthreads(); \
    for (unsigned int stride = lossThreadCountCUDA / 2u; stride > 0u; stride >>= 1u) { \
        if (threadIndex < stride) { \
            reduction[threadIndex] = fmaxf(reduction[threadIndex], reduction[threadIndex + stride]); \
        } \
        __syncthreads(); \
    } \
    float maximum = reduction[0]; \
    float localSum = 0.0f; \
    for (unsigned int col = threadIndex; targetOK && col < classes; col += lossThreadCountCUDA) { \
        localSum += expf(loadFn(logits, rowOffset + col) - maximum); \
    } \
    reduction[threadIndex] = localSum; \
    __syncthreads(); \
    for (unsigned int stride = lossThreadCountCUDA / 2u; stride > 0u; stride >>= 1u) { \
        if (threadIndex < stride) { \
            reduction[threadIndex] += reduction[threadIndex + stride]; \
        } \
        __syncthreads(); \
    } \
    if (threadIndex == 0u) { \
        float targetLogit = targetOK ? loadFn(logits, rowOffset + static_cast<unsigned int>(targetID)) : 0.0f; \
        scratch[rowIndex] = targetOK ? -(targetLogit - maximum - logf(reduction[0])) : 0.0f; \
    } \
}

CROSS_ENTROPY_PARTIAL_KERNEL(cross_entropy_float32, float, loss_load_f32)
CROSS_ENTROPY_PARTIAL_KERNEL(cross_entropy_float16, __half, loss_load_f16)
CROSS_ENTROPY_PARTIAL_KERNEL(cross_entropy_bfloat16, __nv_bfloat16, loss_load_bf16)

#endif
