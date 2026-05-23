#include <metal_stdlib>

using namespace metal;

constant uint samplingThreadCount = 256;
constant uint samplingInvalidIndex = 0xffffffffu;

static inline float sampling_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

struct Float32SamplingStorage {
    static float load(device const float* values, uint index) {
        return values[index];
    }
};

struct Float16SamplingStorage {
    static float load(device const half* values, uint index) {
        return float(values[index]);
    }
};

struct BFloat16SamplingStorage {
    static float load(device const ushort* values, uint index) {
        return sampling_bf16_to_float(values[index]);
    }
};

static inline bool sampling_candidate_before(
    float leftScore,
    uint leftIndex,
    float rightScore,
    uint rightIndex
) {
    if (leftScore > rightScore) {
        return true;
    }

    if (leftScore < rightScore) {
        return false;
    }

    return leftIndex < rightIndex;
}

template <typename Storage, typename Scalar>
static inline void greedy_sample_kernel(
    device const Scalar* logits,
    device int* out,
    threadgroup float* scoreScratch,
    threadgroup uint* indexScratch,
    constant uint& count,
    uint threadIndex
) {
    float bestScore = -3.4028234663852886e38f;
    uint bestIndex = samplingInvalidIndex;

    for (uint index = threadIndex; index < count; index += samplingThreadCount) {
        float score = Storage::load(logits, index);
        if (sampling_candidate_before(score, index, bestScore, bestIndex)) {
            bestScore = score;
            bestIndex = index;
        }
    }

    scoreScratch[threadIndex] = bestScore;
    indexScratch[threadIndex] = bestIndex;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = samplingThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            float rightScore = scoreScratch[threadIndex + stride];
            uint rightIndex = indexScratch[threadIndex + stride];
            if (sampling_candidate_before(rightScore, rightIndex, scoreScratch[threadIndex], indexScratch[threadIndex])) {
                scoreScratch[threadIndex] = rightScore;
                indexScratch[threadIndex] = rightIndex;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (threadIndex == 0) {
        out[0] = int(indexScratch[0]);
    }
}

template <typename Storage, typename Scalar>
static inline void sampling_init_kernel(
    device const Scalar* logits,
    device float* scores,
    device uint* indices,
    constant uint& count,
    constant uint& paddedCount,
    uint index
) {
    if (index >= paddedCount) {
        return;
    }

    if (index < count) {
        scores[index] = Storage::load(logits, index);
        indices[index] = index;
        return;
    }

    scores[index] = -3.4028234663852886e38f;
    indices[index] = samplingInvalidIndex;
}
