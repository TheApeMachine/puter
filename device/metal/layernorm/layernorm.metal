#include "layernorm_common.metalinc"
#include "layernorm_apply.metalinc"

#include <metal_stdlib>

using namespace metal;

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
    float localSquareCompensation = 0.0f;

    for (uint col = threadIndex; col < cols; col += normalizationThreadCount) {
        float value = Storage::load(input, rowOffset + col);
        float square = value * value;
        float compensated = square - localSquareCompensation;
        float nextSum = localSquareSum + compensated;
        localSquareCompensation = (nextSum - localSquareSum) - compensated;
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
    constant uint& rowsPerBatch,
    constant uint& modulationCols,
    constant float& epsilon,
    uint row,
    uint threadIndex
) {
    uint rowOffset = row * cols;
    uint batch = row / rowsPerBatch;
    uint modulationOffset = batch * modulationCols;
    float localSquareSum = 0.0f;
    float localSquareCompensation = 0.0f;

    for (uint col = threadIndex; col < cols; col += normalizationThreadCount) {
        float value = Storage::load(input, rowOffset + col);
        float square = value * value;
        float compensated = square - localSquareCompensation;
        float nextSum = localSquareSum + compensated;
        localSquareCompensation = (nextSum - localSquareSum) - compensated;
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

    float invRMS = rsqrt(reduction[0] / float(cols) + epsilon);

    for (uint col = threadIndex; col < cols; col += normalizationThreadCount) {
        float normalized = Storage::load(input, rowOffset + col) * invRMS;
        float scale = Storage::load(modulation, modulationOffset + col);
        float shift = Storage::load(modulation, modulationOffset + cols + col);
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
    constant float& epsilon,
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

    float invStdDev = rsqrt(reduction[0] / float(cols) + epsilon);

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
