#include "embedding.cuh"

static __device__ __forceinline__ float timestep_embedding_value(
    const float* timesteps,
    const float* maxPeriod,
    const float* downscaleFreqShift,
    const int* flipSinToCos,
    unsigned int dim,
    unsigned int index
) {
    unsigned int halfDim = dim / 2u;
    unsigned int row = index / dim;
    unsigned int column = index - row * dim;

    if (halfDim == 0u || column >= halfDim * 2u) {
        return 0.0f;
    }

    bool flipped = flipSinToCos[0] != 0;
    bool firstHalf = column < halfDim;
    unsigned int frequencyIndex = firstHalf ? column : column - halfDim;
    float denominator = static_cast<float>(halfDim) - downscaleFreqShift[0];
    float exponent = -logf(maxPeriod[0]) * static_cast<float>(frequencyIndex) / denominator;
    float angle = timesteps[row] * expf(exponent);
    float sinValue = sinf(angle);
    float cosValue = cosf(angle);

    if (flipped) {
        return firstHalf ? cosValue : sinValue;
    }

    return firstHalf ? sinValue : cosValue;
}

static __device__ __forceinline__ void timestep_store_f32(float* out, unsigned int index, float value) {
    out[index] = value;
}

static __device__ __forceinline__ void timestep_store_f16(__half* out, unsigned int index, float value) {
    out[index] = __float2half(value);
}

static __device__ __forceinline__ void timestep_store_bf16(__nv_bfloat16* out, unsigned int index, float value) {
    out[index] = __float2bfloat16(value);
}

#define TIMESTEP_EMBEDDING_KERNEL(name, scalarType, storeFn) \
extern "C" __global__ void name( \
    const float* timesteps, \
    const float* maxPeriod, \
    const float* downscaleFreqShift, \
    const int* flipSinToCos, \
    scalarType* out, \
    unsigned int count, \
    unsigned int dim \
) { \
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int total = count * dim; \
    if (index >= total) { \
        return; \
    } \
    storeFn( \
        out, \
        index, \
        timestep_embedding_value(timesteps, maxPeriod, downscaleFreqShift, flipSinToCos, dim, index) \
    ); \
}

TIMESTEP_EMBEDDING_KERNEL(timestep_embedding_float32, float, timestep_store_f32)
TIMESTEP_EMBEDDING_KERNEL(timestep_embedding_float16, __half, timestep_store_f16)
TIMESTEP_EMBEDDING_KERNEL(timestep_embedding_bfloat16, __nv_bfloat16, timestep_store_bf16)
