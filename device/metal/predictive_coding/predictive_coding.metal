#include <metal_stdlib>

using namespace metal;

constant float predictiveCodingLearningRate = 1.0e-2f;

template <typename Storage, typename Scalar>
static inline void pc_prediction_kernel(
    device const Scalar* weights,
    device const Scalar* state,
    device Scalar* out,
    constant uint& inCount,
    uint outIndex
) {
    float sum = 0.0f;
    uint rowOffset = outIndex * inCount;

    for (uint inIndex = 0; inIndex < inCount; inIndex++) {
        sum += Storage::load(weights, rowOffset + inIndex) * Storage::load(state, inIndex);
    }

    Storage::store(out, outIndex, sum);
}

template <typename Storage, typename Scalar>
static inline void pc_prediction_error_kernel(
    device const Scalar* observed,
    device const Scalar* predicted,
    device Scalar* out,
    constant uint& count,
    uint index
) {
    if (index >= count) {
        return;
    }

    Storage::store(out, index, Storage::load(observed, index) - Storage::load(predicted, index));
}

template <typename Storage, typename Scalar>
static inline void pc_update_representation_kernel(
    device const Scalar* weights,
    device const Scalar* state,
    device const Scalar* predictionError,
    device Scalar* out,
    constant uint& outCount,
    constant uint& inCount,
    uint inIndex
) {
    if (inIndex >= inCount) {
        return;
    }

    float value = Storage::load(state, inIndex);

    for (uint outIndex = 0; outIndex < outCount; outIndex++) {
        value += predictiveCodingLearningRate *
            Storage::load(weights, outIndex * inCount + inIndex) *
            Storage::load(predictionError, outIndex);
    }

    Storage::store(out, inIndex, value);
}

template <typename Storage, typename Scalar>
static inline void pc_update_weights_kernel(
    device const Scalar* weights,
    device const Scalar* state,
    device const Scalar* predictionError,
    device Scalar* out,
    constant uint& inCount,
    constant uint& count,
    uint index
) {
    if (index >= count) {
        return;
    }

    uint outIndex = index / inCount;
    uint inIndex = index - outIndex * inCount;
    float value = Storage::load(weights, index) +
        predictiveCodingLearningRate *
        Storage::load(predictionError, outIndex) *
        Storage::load(state, inIndex);

    Storage::store(out, index, value);
}

