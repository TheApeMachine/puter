#include "sampling.metal"

using namespace metal;

SAMPLING_INIT_KERNEL(sampling_init_float32, Float32SamplingStorage, float)
SAMPLING_INIT_KERNEL(sampling_init_float16, Float16SamplingStorage, half)
SAMPLING_INIT_KERNEL(sampling_init_bfloat16, BFloat16SamplingStorage, ushort)

kernel void sampling_bitonic_step(
    device float* scores [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    constant uint& stageSize [[buffer(2)]],
    constant uint& passSize [[buffer(3)]],
    constant uint& paddedCount [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= paddedCount) {
        return;
    }

    uint peer = index ^ passSize;
    if (peer <= index || peer >= paddedCount) {
        return;
    }

    float leftScore = scores[index];
    uint leftIndex = indices[index];
    float rightScore = scores[peer];
    uint rightIndex = indices[peer];
    bool descending = (index & stageSize) == 0;
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

kernel void sampling_draw_sorted(
    device const float* scores [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device int* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant float& target [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    if (index != 0) {
        return;
    }

    float maximum = scores[0];
    float denominator = 0.0f;

    for (uint scoreIndex = 0; scoreIndex < count; scoreIndex++) {
        denominator += exp(scores[scoreIndex] - maximum);
    }

    float targetMass = target * denominator;
    float cumulative = 0.0f;
    uint chosen = indices[count - 1];

    for (uint scoreIndex = 0; scoreIndex < count; scoreIndex++) {
        cumulative += exp(scores[scoreIndex] - maximum);

        if (cumulative >= targetMass) {
            chosen = indices[scoreIndex];
            break;
        }
    }

    out[0] = int(chosen);
}
