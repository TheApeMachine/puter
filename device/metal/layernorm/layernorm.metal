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
static inline void layernorm_rows(
    device const Scalar* input,
    device const Scalar* scale,
    device const Scalar* bias,
    device Scalar* out,
    threadgroup float* reduction,
    constant uint& cols,
    uint row,
    uint threadIndex
) {
    uint rowOffset = row * cols;
    float mean = reduce_sum<Storage, Scalar>(input, reduction, rowOffset, cols, threadIndex) /
        float(cols);
    float localVariance = 0.0f;
    float localCompensation = 0.0f;

    for (uint col = threadIndex; col < cols; col += normalizationThreadCount) {
        float delta = Storage::load(input, rowOffset + col) - mean;
        float value = delta * delta - localCompensation;
        float nextVariance = localVariance + value;
        localCompensation = (nextVariance - localVariance) - value;
        localVariance = nextVariance;
    }

    reduction[threadIndex] = localVariance;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = normalizationThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            reduction[threadIndex] += reduction[threadIndex + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float invStdDev = 1.0f / sqrt(reduction[0] / float(cols) + layerNormEpsilonMetal);

    for (uint col = threadIndex; col < cols; col += normalizationThreadCount) {
        float normalized = (Storage::load(input, rowOffset + col) - mean) * invStdDev;
        float value = normalized * Storage::load(scale, col) + Storage::load(bias, col);
        Storage::store(out, rowOffset + col, value);
    }
}

template <typename Storage, typename Scalar>
static inline void rmsnorm_rows(
    device const Scalar* input,
    device const Scalar* scale,
    device Scalar* out,
    threadgroup float* reduction,
    constant uint& cols,
    constant float& epsilon,
    uint row,
    uint threadIndex
) {
    uint rowOffset = row * cols;
    float localSquareSum = 0.0f;
    float localCompensation = 0.0f;

    for (uint col = threadIndex; col < cols; col += normalizationThreadCount) {
        float value = Storage::load(input, rowOffset + col);
        float square = value * value - localCompensation;
        float nextSum = localSquareSum + square;
        localCompensation = (nextSum - localSquareSum) - square;
        localSquareSum = nextSum;
    }

    reduction[threadIndex] = localSquareSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = normalizationThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            reduction[threadIndex] += reduction[threadIndex + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float invRMS = 1.0f / sqrt(reduction[0] / float(cols) + epsilon);

    for (uint col = threadIndex; col < cols; col += normalizationThreadCount) {
        float value = Storage::load(input, rowOffset + col) * invRMS * Storage::load(scale, col);
        Storage::store(out, rowOffset + col, value);
    }
}

template <typename Storage, typename Scalar>
static inline void adaptive_rmsnorm_rows(
    device const Scalar* input,
    device const Scalar* modulation,
    device Scalar* out,
    threadgroup float* reduction,
    constant uint& cols,
    uint row,
    uint threadIndex
) {
    uint rowOffset = row * cols;
    float localSquareSum = 0.0f;
    float localCompensation = 0.0f;

    for (uint col = threadIndex; col < cols; col += normalizationThreadCount) {
        float value = Storage::load(input, rowOffset + col);
        float square = value * value - localCompensation;
        float nextSum = localSquareSum + square;
        localCompensation = (nextSum - localSquareSum) - square;
        localSquareSum = nextSum;
    }

    reduction[threadIndex] = localSquareSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = normalizationThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            reduction[threadIndex] += reduction[threadIndex + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float invRMS = 1.0f / sqrt(reduction[0] / float(cols) + rmsNormEpsilonMetalDefault);

    for (uint col = threadIndex; col < cols; col += normalizationThreadCount) {
        float normalized = Storage::load(input, rowOffset + col) * invRMS;
        float scale = Storage::load(modulation, col);
        float shift = Storage::load(modulation, cols + col);
        Storage::store(out, rowOffset + col, normalized * (1.0f + scale) + shift);
    }
}

template <typename Storage, typename Scalar>
static inline void modulated_layernorm_rows(
    device const Scalar* input,
    device const Scalar* modulation,
    device Scalar* out,
    threadgroup float* reduction,
    constant uint& cols,
    constant uint& rowsPerBatch,
    constant uint& modulationCols,
    constant uint& modulationSet,
    uint row,
    uint threadIndex
) {
    uint rowOffset = row * cols;
    uint batch = row / rowsPerBatch;
    uint modulationOffset = batch * modulationCols + modulationSet * cols * 3;
    float mean = reduce_sum<Storage, Scalar>(input, reduction, rowOffset, cols, threadIndex) /
        float(cols);
    float localVariance = 0.0f;
    float localCompensation = 0.0f;

    for (uint col = threadIndex; col < cols; col += normalizationThreadCount) {
        float delta = Storage::load(input, rowOffset + col) - mean;
        float value = delta * delta - localCompensation;
        float nextVariance = localVariance + value;
        localCompensation = (nextVariance - localVariance) - value;
        localVariance = nextVariance;
    }

    reduction[threadIndex] = localVariance;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = normalizationThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            reduction[threadIndex] += reduction[threadIndex + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float invStdDev = 1.0f / sqrt(reduction[0] / float(cols) + layerNormEpsilonMetal);

    for (uint col = threadIndex; col < cols; col += normalizationThreadCount) {
        float normalized = (Storage::load(input, rowOffset + col) - mean) * invStdDev;
        float shift = Storage::load(modulation, modulationOffset + col);
        float scale = Storage::load(modulation, modulationOffset + cols + col);
        Storage::store(out, rowOffset + col, normalized * (1.0f + scale) + shift);
    }
}

template <typename Storage, typename Scalar>
static inline void gated_residual_values(
    device const Scalar* residual,
    device const Scalar* branch,
    device const Scalar* modulation,
    device Scalar* out,
    constant uint& cols,
    constant uint& rowsPerBatch,
    constant uint& modulationCols,
    constant uint& modulationSet,
    uint row,
    uint threadIndex
) {
    uint batch = row / rowsPerBatch;
    uint modulationOffset = batch * modulationCols + modulationSet * cols * 3 + cols * 2;
    uint rowOffset = row * cols;

    for (uint col = threadIndex; col < cols; col += normalizationThreadCount) {
        uint index = rowOffset + col;
        float gate = Storage::load(modulation, modulationOffset + col);
        float value = Storage::load(residual, index) + gate * Storage::load(branch, index);
        Storage::store(out, index, value);
    }
}

template <typename Storage, typename Scalar>
