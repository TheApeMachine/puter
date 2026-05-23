#include <metal_stdlib>

using namespace metal;

constant float predictiveCodingLearningRate = 1.0e-2f;

static inline float research_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort research_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

struct Float32ResearchStorage {
    static float load(device const float* values, uint index) {
        return values[index];
    }

    static void store(device float* values, uint index, float value) {
        values[index] = value;
    }
};

struct Float16ResearchStorage {
    static float load(device const half* values, uint index) {
        return float(values[index]);
    }

    static void store(device half* values, uint index, float value) {
        values[index] = half(value);
    }
};

struct BFloat16ResearchStorage {
    static float load(device const ushort* values, uint index) {
        return research_bf16_to_float(values[index]);
    }

    static void store(device ushort* values, uint index, float value) {
        values[index] = research_float_to_bf16(value);
    }
};

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

#define RESEARCH_BINARY_KERNEL(name, body, storage, scalar) \
kernel void name( \
    device const scalar* left [[buffer(0)]], \
    device const scalar* right [[buffer(1)]], \
    device scalar* out [[buffer(2)]], \
    constant uint& count [[buffer(3)]], \
    uint index [[thread_position_in_grid]] \
) { \
    body<storage, scalar>(left, right, out, count, index); \
}

#define RESEARCH_UNARY_KERNEL(name, body, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device scalar* out [[buffer(1)]], \
    constant uint& count [[buffer(2)]], \
    uint index [[thread_position_in_grid]] \
) { \
    body<storage, scalar>(input, out, count, index); \
}

#define PC_PREDICTION_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* weights [[buffer(0)]], \
    device const scalar* state [[buffer(1)]], \
    device scalar* out [[buffer(2)]], \
    constant uint& inCount [[buffer(3)]], \
    uint outIndex [[thread_position_in_grid]] \
) { \
    pc_prediction_kernel<storage, scalar>(weights, state, out, inCount, outIndex); \
}

#define PC_UPDATE_REPRESENTATION_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* weights [[buffer(0)]], \
    device const scalar* state [[buffer(1)]], \
    device const scalar* predictionError [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    constant uint& outCount [[buffer(4)]], \
    constant uint& inCount [[buffer(5)]], \
    uint inIndex [[thread_position_in_grid]] \
) { \
    pc_update_representation_kernel<storage, scalar>( \
        weights, state, predictionError, out, outCount, inCount, inIndex \
    ); \
}

#define PC_UPDATE_WEIGHTS_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* weights [[buffer(0)]], \
    device const scalar* state [[buffer(1)]], \
    device const scalar* predictionError [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    constant uint& inCount [[buffer(4)]], \
    constant uint& count [[buffer(5)]], \
    uint index [[thread_position_in_grid]] \
) { \
    pc_update_weights_kernel<storage, scalar>( \
        weights, state, predictionError, out, inCount, count, index \
    ); \
}
