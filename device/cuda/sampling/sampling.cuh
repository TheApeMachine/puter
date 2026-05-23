#ifndef PUTER_DEVICE_CUDA_SAMPLING_SAMPLING_CUH
#define PUTER_DEVICE_CUDA_SAMPLING_SAMPLING_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

static constexpr unsigned int samplingThreadCountCUDA = 256u;
static constexpr unsigned int samplingInvalidIndexCUDA = 0xffffffffu;

static __device__ __forceinline__ float sampling_load_f32(const float* values, unsigned int index) {
    return values[index];
}

static __device__ __forceinline__ float sampling_load_f16(const __half* values, unsigned int index) {
    return __half2float(values[index]);
}

static __device__ __forceinline__ float sampling_load_bf16(const __nv_bfloat16* values, unsigned int index) {
    return __bfloat162float(values[index]);
}

static __device__ __forceinline__ bool sampling_candidate_before(
    float leftScore,
    unsigned int leftIndex,
    float rightScore,
    unsigned int rightIndex
) {
    if (leftScore > rightScore) {
        return true;
    }

    if (leftScore < rightScore) {
        return false;
    }

    return leftIndex < rightIndex;
}

#define GREEDY_SAMPLE_KERNEL(name, scalarType, loadFn) \
extern "C" __global__ void name( \
    const scalarType* logits, \
    int* out, \
    unsigned int count \
) { \
    __shared__ float scoreScratch[samplingThreadCountCUDA]; \
    __shared__ unsigned int indexScratch[samplingThreadCountCUDA]; \
    unsigned int threadIndex = threadIdx.x; \
    float bestScore = -CUDART_INF_F; \
    unsigned int bestIndex = samplingInvalidIndexCUDA; \
    for (unsigned int index = threadIndex; index < count; index += samplingThreadCountCUDA) { \
        float score = loadFn(logits, index); \
        if (sampling_candidate_before(score, index, bestScore, bestIndex)) { \
            bestScore = score; \
            bestIndex = index; \
        } \
    } \
    scoreScratch[threadIndex] = bestScore; \
    indexScratch[threadIndex] = bestIndex; \
    __syncthreads(); \
    for (unsigned int stride = samplingThreadCountCUDA / 2u; stride > 0u; stride >>= 1u) { \
        if (threadIndex < stride) { \
            float rightScore = scoreScratch[threadIndex + stride]; \
            unsigned int rightIndex = indexScratch[threadIndex + stride]; \
            if (sampling_candidate_before( \
                rightScore, rightIndex, scoreScratch[threadIndex], indexScratch[threadIndex] \
            )) { \
                scoreScratch[threadIndex] = rightScore; \
                indexScratch[threadIndex] = rightIndex; \
            } \
        } \
        __syncthreads(); \
    } \
    if (threadIndex == 0u) { \
        out[0] = static_cast<int>(indexScratch[0]); \
    } \
}

#define SAMPLING_INIT_KERNEL(name, scalarType, loadFn) \
extern "C" __global__ void name( \
    const scalarType* logits, \
    float* scores, \
    unsigned int* indices, \
    unsigned int count, \
    unsigned int paddedCount \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    if (index >= paddedCount) { \
        return; \
    } \
    if (index < count) { \
        scores[index] = loadFn(logits, index); \
        indices[index] = index; \
        return; \
    } \
    scores[index] = -CUDART_INF_F; \
    indices[index] = samplingInvalidIndexCUDA; \
}

extern "C" __global__ void sampling_bitonic_step(
    float* scores,
    unsigned int* indices,
    unsigned int stageSize,
    unsigned int passSize,
    unsigned int paddedCount
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= paddedCount) {
        return;
    }

    unsigned int peer = index ^ passSize;

    if (peer <= index || peer >= paddedCount) {
        return;
    }

    float leftScore = scores[index];
    unsigned int leftIndex = indices[index];
    float rightScore = scores[peer];
    unsigned int rightIndex = indices[peer];
    bool descending = (index & stageSize) == 0u;
    bool rightBeforeLeft = sampling_candidate_before(rightScore, rightIndex, leftScore, leftIndex);
    bool leftBeforeRight = sampling_candidate_before(leftScore, leftIndex, rightScore, rightIndex);

    if (descending && !rightBeforeLeft) {
        return;
    }

    if (!descending && !leftBeforeRight) {
        return;
    }

    scores[index] = rightScore;
    indices[index] = rightIndex;
    scores[peer] = leftScore;
    indices[peer] = leftIndex;
}

extern "C" __global__ void sampling_draw_sorted(
    const float* scores,
    const unsigned int* indices,
    int* out,
    unsigned int count,
    float target
) {
    if (threadIdx.x != 0u || blockIdx.x != 0u) {
        return;
    }

    float maximum = scores[0];
    float denominator = 0.0f;

    for (unsigned int scoreIndex = 0u; scoreIndex < count; scoreIndex++) {
        denominator += expf(scores[scoreIndex] - maximum);
    }

    float targetMass = target * denominator;
    float cumulative = 0.0f;
    unsigned int chosen = indices[count - 1u];

    for (unsigned int scoreIndex = 0u; scoreIndex < count; scoreIndex++) {
        cumulative += expf(scores[scoreIndex] - maximum);

        if (cumulative >= targetMass) {
            chosen = indices[scoreIndex];
            break;
        }
    }

    out[0] = static_cast<int>(chosen);
}

GREEDY_SAMPLE_KERNEL(greedy_sample_float32, float, sampling_load_f32)
GREEDY_SAMPLE_KERNEL(greedy_sample_float16, __half, sampling_load_f16)
GREEDY_SAMPLE_KERNEL(greedy_sample_bfloat16, __nv_bfloat16, sampling_load_bf16)

SAMPLING_INIT_KERNEL(sampling_init_float32, float, sampling_load_f32)
SAMPLING_INIT_KERNEL(sampling_init_float16, __half, sampling_load_f16)
SAMPLING_INIT_KERNEL(sampling_init_bfloat16, __nv_bfloat16, sampling_load_bf16)

#endif
