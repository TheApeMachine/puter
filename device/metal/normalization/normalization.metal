#include <metal_stdlib>

using namespace metal;

constant uint normalizationThreadCount = 256;
constant float layerNormEpsilonMetal = 1.0e-5f;
constant float rmsNormEpsilonMetalDefault = 1.0e-6f;

static inline float bf16_to_float_norm(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort float_to_bf16_norm(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

struct Float32NormStorage {
    static float load(device const float* values, uint index) {
        return values[index];
    }

    static void store(device float* values, uint index, float value) {
        values[index] = value;
    }
};

struct Float16NormStorage {
    static float load(device const half* values, uint index) {
        return float(values[index]);
    }

    static void store(device half* values, uint index, float value) {
        values[index] = half(value);
    }
};

struct BFloat16NormStorage {
    static float load(device const ushort* values, uint index) {
        return bf16_to_float_norm(values[index]);
    }

    static void store(device ushort* values, uint index, float value) {
        values[index] = float_to_bf16_norm(value);
    }
};

template <typename Storage, typename Scalar>
static inline float reduce_sum(
    device const Scalar* input,
    threadgroup float* reduction,
    uint rowOffset,
    uint cols,
    uint threadIndex
) {
    float localSum = 0.0f;
    float localCompensation = 0.0f;

    for (uint col = threadIndex; col < cols; col += normalizationThreadCount) {
        float value = Storage::load(input, rowOffset + col) - localCompensation;
        float nextSum = localSum + value;
        localCompensation = (nextSum - localSum) - value;
        localSum = nextSum;
    }

    reduction[threadIndex] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = normalizationThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            reduction[threadIndex] += reduction[threadIndex + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    return reduction[0];
}

template <typename Storage, typename Scalar>
