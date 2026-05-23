#include "embedding.metal"

using namespace metal;



static inline ushort timestep_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

static inline float timestep_embedding_value(
    device const float* timesteps,
    device const float* maxPeriod,
    device const float* downscaleFreqShift,
    device const int* flipSinToCos,
    uint dim,
    uint index
) {
    uint halfDim = dim / 2u;
    uint row = index / dim;
    uint column = index - row * dim;

    if (halfDim == 0u || column >= halfDim * 2u) {
        return 0.0f;
    }

    bool flipped = flipSinToCos[0] != 0;
    bool firstHalf = column < halfDim;
    uint frequencyIndex = firstHalf ? column : column - halfDim;
    float denominator = float(halfDim) - downscaleFreqShift[0];
    float exponent = -log(maxPeriod[0]) * float(frequencyIndex) / denominator;
    float angle = timesteps[row] * exp(exponent);
    float sinValue = sin(angle);
    float cosValue = cos(angle);

    if (flipped) {
        return firstHalf ? cosValue : sinValue;
    }

    return firstHalf ? sinValue : cosValue;
}

kernel void timestep_embedding_float32(
    device const float* timesteps [[buffer(0)]],
    device const float* maxPeriod [[buffer(1)]],
    device const float* downscaleFreqShift [[buffer(2)]],
    device const int* flipSinToCos [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    constant uint& dim [[buffer(6)]],
    uint index [[thread_position_in_grid]]
) {
    uint total = count * dim;

    if (index >= total) {
        return;
    }

    out[index] = timestep_embedding_value(
        timesteps, maxPeriod, downscaleFreqShift, flipSinToCos, dim, index
    );
}

kernel void timestep_embedding_float16(
    device const float* timesteps [[buffer(0)]],
    device const float* maxPeriod [[buffer(1)]],
    device const float* downscaleFreqShift [[buffer(2)]],
    device const int* flipSinToCos [[buffer(3)]],
    device half* out [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    constant uint& dim [[buffer(6)]],
    uint index [[thread_position_in_grid]]
) {
    uint total = count * dim;

    if (index >= total) {
        return;
    }

    out[index] = half(timestep_embedding_value(
        timesteps, maxPeriod, downscaleFreqShift, flipSinToCos, dim, index
    ));
}

kernel void timestep_embedding_bfloat16(
    device const float* timesteps [[buffer(0)]],
    device const float* maxPeriod [[buffer(1)]],
    device const float* downscaleFreqShift [[buffer(2)]],
    device const int* flipSinToCos [[buffer(3)]],
    device ushort* out [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    constant uint& dim [[buffer(6)]],
    uint index [[thread_position_in_grid]]
) {
    uint total = count * dim;

    if (index >= total) {
        return;
    }

    float value = timestep_embedding_value(
        timesteps, maxPeriod, downscaleFreqShift, flipSinToCos, dim, index
    );
    out[index] = timestep_float_to_bf16(value);
}
