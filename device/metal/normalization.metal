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

static inline float refined_inv_sqrt_norm(float value) {
    float estimate = 1.0f / precise::sqrt(value);
    float halfValue = 0.5f * value;
    estimate = estimate * (1.5f - halfValue * estimate * estimate);
    return estimate;
}

static inline float2 ds_renorm(float high, float low) {
    float sum = high + low;
    float error = low - (sum - high);
    return float2(sum, error);
}

static inline float2 ds_add_float(float2 value, float addend) {
    float sum = value.x + addend;
    float virtualAddend = sum - value.x;
    float error = (value.x - (sum - virtualAddend)) + (addend - virtualAddend);
    return ds_renorm(sum, value.y + error);
}

static inline float2 ds_add_pair(float2 left, float2 right) {
    float2 withHigh = ds_add_float(left, right.x);
    return ds_add_float(withHigh, right.y);
}

static inline float2 ds_neg(float2 value) {
    return float2(-value.x, -value.y);
}

static inline float2 ds_sub_from_float(float value, float2 subtrahend) {
    return ds_add_float(ds_neg(subtrahend), value);
}

static inline float2 ds_mul_pair(float2 left, float2 right) {
    float product = left.x * right.x;
    float error = fma(left.x, right.x, -product) + left.x * right.y + left.y * right.x;
    return ds_renorm(product, error);
}

static inline float2 ds_mul_float(float2 value, float scalar) {
    float product = value.x * scalar;
    float error = fma(value.x, scalar, -product) + value.y * scalar;
    return ds_renorm(product, error);
}

static inline float2 ds_div_count(float2 value, uint count) {
    return ds_mul_float(value, 1.0f / float(count));
}

static inline float ds_to_float(float2 value) {
    return value.x + value.y;
}

static inline float ds_inv_sqrt(float2 value, float epsilon) {
    float high = value.x + epsilon;
    float estimate = refined_inv_sqrt_norm(high);
    return estimate * (1.0f - 0.5f * value.y / high);
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
static inline void batchnorm_denorm_values(
    device const Scalar* input,
    device const Scalar* mean,
    device const Scalar* variance,
    device Scalar* out,
    constant uint& channels,
    constant uint& spatial,
    uint row,
    uint threadIndex
) {
    uint channel = row % channels;
    float channelMean = Storage::load(mean, channel);
    float channelStdDev = sqrt(Storage::load(variance, channel) + layerNormEpsilonMetal);
    uint offset = row * spatial;

    for (uint index = threadIndex; index < spatial; index += normalizationThreadCount) {
        float value = Storage::load(input, offset + index);
        Storage::store(out, offset + index, value * channelStdDev + channelMean);
    }
}

template <typename Storage, typename Scalar>
static inline void groupnorm_rows(
    device const Scalar* input,
    device const Scalar* scale,
    device const Scalar* bias,
    device Scalar* out,
    threadgroup float* reduction,
    constant uint& channels,
    constant uint& spatial,
    constant uint& groups,
    uint row,
    uint threadIndex
) {
    uint groupIndex = row % groups;
    uint batchIndex = row / groups;
    uint channelsPerGroup = channels / groups;
    uint channelStart = groupIndex * channelsPerGroup;
    uint groupSize = channelsPerGroup * spatial;
    uint groupOffset = (batchIndex * channels + channelStart) * spatial;
    float mean = reduce_sum<Storage, Scalar>(input, reduction, groupOffset, groupSize, threadIndex) /
        float(groupSize);
    float localVariance = 0.0f;
    float localCompensation = 0.0f;

    for (uint offset = threadIndex; offset < groupSize; offset += normalizationThreadCount) {
        float delta = Storage::load(input, groupOffset + offset) - mean;
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

    float invStdDev = 1.0f / precise::sqrt(reduction[0] / float(groupSize) + layerNormEpsilonMetal);

    for (uint offset = threadIndex; offset < groupSize; offset += normalizationThreadCount) {
        uint channel = channelStart + offset / spatial;
        float centered = Storage::load(input, groupOffset + offset) - mean;
        float value = fma(
            centered * invStdDev,
            Storage::load(scale, channel),
            Storage::load(bias, channel)
        );
        Storage::store(out, groupOffset + offset, value);
    }
}

template <typename Storage, typename Scalar>
static inline void instancenorm_rows(
    device const Scalar* input,
    device const Scalar* scale,
    device const Scalar* bias,
    device Scalar* out,
    threadgroup float* reduction,
    constant uint& channels,
    constant uint& spatial,
    uint row,
    uint threadIndex
) {
    uint channel = row % channels;
    uint rowOffset = row * spatial;
    if (threadIndex == 0) {
        float sum = 0.0f;

        for (uint offset = 0; offset < spatial; offset++) {
            sum += Storage::load(input, rowOffset + offset);
        }

        float mean = sum / float(spatial);
        float variance = 0.0f;

        for (uint offset = 0; offset < spatial; offset++) {
            float delta = Storage::load(input, rowOffset + offset) - mean;
            variance += delta * delta;
        }

        reduction[0] = mean;
        reduction[1] = 1.0f / sqrt(variance / float(spatial) + layerNormEpsilonMetal);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean = reduction[0];
    float invStdDev = reduction[1];

    for (uint offset = threadIndex; offset < spatial; offset += normalizationThreadCount) {
        float centered = Storage::load(input, rowOffset + offset) - mean;
        float value = fma(
            centered * invStdDev,
            Storage::load(scale, channel),
            Storage::load(bias, channel)
        );
        Storage::store(out, rowOffset + offset, value);
    }
}

template <typename Storage, typename Scalar>
static inline void batchnorm_eval_rows(
    device const Scalar* input,
    device const Scalar* scale,
    device const Scalar* bias,
    device const Scalar* mean,
    device const Scalar* variance,
    device Scalar* out,
    constant uint& channels,
    constant uint& spatial,
    uint row,
    uint threadIndex
) {
    uint channel = row % channels;
    uint rowOffset = row * spatial;
    float invStdDev = 1.0f / precise::sqrt(Storage::load(variance, channel) + layerNormEpsilonMetal);
    float channelMean = Storage::load(mean, channel);
    float channelScale = Storage::load(scale, channel);
    float channelBias = Storage::load(bias, channel);

    for (uint offset = threadIndex; offset < spatial; offset += normalizationThreadCount) {
        float centered = Storage::load(input, rowOffset + offset) - channelMean;
        Storage::store(out, rowOffset + offset, fma(centered * invStdDev, channelScale, channelBias));
    }
}

#define LAYERNORM_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device const scalar* scale [[buffer(1)]], \
    device const scalar* bias [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    constant uint& cols [[buffer(4)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    threadgroup float reduction[256]; \
    layernorm_rows<storage, scalar>(input, scale, bias, out, reduction, cols, row, threadIndex); \
}

#define RMSNORM_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device const scalar* scale [[buffer(1)]], \
    device scalar* out [[buffer(2)]], \
    constant uint& cols [[buffer(3)]], \
    constant float& epsilon [[buffer(4)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    threadgroup float reduction[256]; \
    rmsnorm_rows<storage, scalar>(input, scale, out, reduction, cols, epsilon, row, threadIndex); \
}

#define ADAPTIVE_RMSNORM_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device const scalar* modulation [[buffer(1)]], \
    device scalar* out [[buffer(2)]], \
    constant uint& cols [[buffer(3)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    threadgroup float reduction[256]; \
    adaptive_rmsnorm_rows<storage, scalar>(input, modulation, out, reduction, cols, row, threadIndex); \
}

#define MODULATED_LAYERNORM_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device const scalar* modulation [[buffer(1)]], \
    device scalar* out [[buffer(2)]], \
    constant uint& cols [[buffer(3)]], \
    constant uint& rowsPerBatch [[buffer(4)]], \
    constant uint& modulationCols [[buffer(5)]], \
    constant uint& modulationSet [[buffer(6)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    threadgroup float reduction[256]; \
    modulated_layernorm_rows<storage, scalar>( \
        input, modulation, out, reduction, cols, rowsPerBatch, modulationCols, modulationSet, row, threadIndex \
    ); \
}

#define GATED_RESIDUAL_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* residual [[buffer(0)]], \
    device const scalar* branch [[buffer(1)]], \
    device const scalar* modulation [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    constant uint& cols [[buffer(4)]], \
    constant uint& rowsPerBatch [[buffer(5)]], \
    constant uint& modulationCols [[buffer(6)]], \
    constant uint& modulationSet [[buffer(7)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    gated_residual_values<storage, scalar>( \
        residual, branch, modulation, out, cols, rowsPerBatch, modulationCols, modulationSet, row, threadIndex \
    ); \
}

#define BATCHNORM_DENORM_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device const scalar* mean [[buffer(1)]], \
    device const scalar* variance [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    constant uint& channels [[buffer(4)]], \
    constant uint& spatial [[buffer(5)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    batchnorm_denorm_values<storage, scalar>(input, mean, variance, out, channels, spatial, row, threadIndex); \
}

#define GROUPNORM_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device const scalar* scale [[buffer(1)]], \
    device const scalar* bias [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    constant uint& channels [[buffer(4)]], \
    constant uint& spatial [[buffer(5)]], \
    constant uint& groups [[buffer(6)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    threadgroup float reduction[256]; \
    groupnorm_rows<storage, scalar>( \
        input, scale, bias, out, reduction, channels, spatial, groups, row, threadIndex \
    ); \
}

#define INSTANCENORM_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device const scalar* scale [[buffer(1)]], \
    device const scalar* bias [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    constant uint& channels [[buffer(4)]], \
    constant uint& spatial [[buffer(5)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    threadgroup float reduction[256]; \
    instancenorm_rows<storage, scalar>( \
        input, scale, bias, out, reduction, channels, spatial, row, threadIndex \
    ); \
}

#define BATCHNORM_EVAL_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device const scalar* scale [[buffer(1)]], \
    device const scalar* bias [[buffer(2)]], \
    device const scalar* mean [[buffer(3)]], \
    device const scalar* variance [[buffer(4)]], \
    device scalar* out [[buffer(5)]], \
    constant uint& channels [[buffer(6)]], \
    constant uint& spatial [[buffer(7)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint threadIndex [[thread_position_in_threadgroup]] \
) { \
    batchnorm_eval_rows<storage, scalar>( \
        input, scale, bias, mean, variance, out, channels, spatial, row, threadIndex \
    ); \
}

kernel void layernorm_float32(
    device const float* input [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_position_in_threadgroup]]
) {
    threadgroup float stats[2];
    uint rowOffset = row * cols;

    if (threadIndex == 0) {
        float sum = 0.0f;

        for (uint col = 0; col < cols; col++) {
            sum += input[rowOffset + col];
        }

        float mean = sum / float(cols);
        float variance = 0.0f;

        for (uint col = 0; col < cols; col++) {
            float delta = input[rowOffset + col] - mean;
            variance += delta * delta;
        }

        stats[0] = mean;
        stats[1] = 1.0f / sqrt(variance / float(cols) + layerNormEpsilonMetal);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean = stats[0];
    float invStdDev = stats[1];

    for (uint col = threadIndex; col < cols; col += normalizationThreadCount) {
        float centered = input[rowOffset + col] - mean;
        out[rowOffset + col] = fma(centered * invStdDev, scale[col], bias[col]);
    }
}

LAYERNORM_KERNEL(layernorm_float16, Float16NormStorage, half)
LAYERNORM_KERNEL(layernorm_bfloat16, BFloat16NormStorage, ushort)

RMSNORM_KERNEL(rmsnorm_float32, Float32NormStorage, float)
RMSNORM_KERNEL(rmsnorm_float16, Float16NormStorage, half)
RMSNORM_KERNEL(rmsnorm_bfloat16, BFloat16NormStorage, ushort)

ADAPTIVE_RMSNORM_KERNEL(adaptive_rmsnorm_float32, Float32NormStorage, float)
ADAPTIVE_RMSNORM_KERNEL(adaptive_rmsnorm_float16, Float16NormStorage, half)
ADAPTIVE_RMSNORM_KERNEL(adaptive_rmsnorm_bfloat16, BFloat16NormStorage, ushort)

MODULATED_LAYERNORM_KERNEL(modulated_layernorm_float32, Float32NormStorage, float)
MODULATED_LAYERNORM_KERNEL(modulated_layernorm_float16, Float16NormStorage, half)
MODULATED_LAYERNORM_KERNEL(modulated_layernorm_bfloat16, BFloat16NormStorage, ushort)

GATED_RESIDUAL_KERNEL(gated_residual_float32, Float32NormStorage, float)
GATED_RESIDUAL_KERNEL(gated_residual_float16, Float16NormStorage, half)
GATED_RESIDUAL_KERNEL(gated_residual_bfloat16, BFloat16NormStorage, ushort)

BATCHNORM_DENORM_KERNEL(batchnorm_denorm_float32, Float32NormStorage, float)
BATCHNORM_DENORM_KERNEL(batchnorm_denorm_float16, Float16NormStorage, half)
BATCHNORM_DENORM_KERNEL(batchnorm_denorm_bfloat16, BFloat16NormStorage, ushort)

kernel void groupnorm_float32(
    device const float* input [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    constant uint& spatial [[buffer(5)]],
    constant uint& groups [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_position_in_threadgroup]]
) {
    threadgroup float stats[2];
    uint groupIndex = row % groups;
    uint batchIndex = row / groups;
    uint channelsPerGroup = channels / groups;
    uint channelStart = groupIndex * channelsPerGroup;
    uint groupSize = channelsPerGroup * spatial;
    uint groupOffset = (batchIndex * channels + channelStart) * spatial;

    if (threadIndex == 0) {
        float sum = 0.0f;

        for (uint offset = 0; offset < groupSize; offset++) {
            sum += input[groupOffset + offset];
        }

        float mean = sum / float(groupSize);
        float variance = 0.0f;

        for (uint offset = 0; offset < groupSize; offset++) {
            float delta = input[groupOffset + offset] - mean;
            variance += delta * delta;
        }

        stats[0] = mean;
        stats[1] = 1.0f / precise::sqrt(variance / float(groupSize) + layerNormEpsilonMetal);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean = stats[0];
    float invStdDev = stats[1];

    for (uint offset = threadIndex; offset < groupSize; offset += normalizationThreadCount) {
        uint channel = channelStart + offset / spatial;
        float centered = input[groupOffset + offset] - mean;
        out[groupOffset + offset] = fma(centered * invStdDev, scale[channel], bias[channel]);
    }
}

GROUPNORM_KERNEL(groupnorm_float16, Float16NormStorage, half)
GROUPNORM_KERNEL(groupnorm_bfloat16, BFloat16NormStorage, ushort)

kernel void instancenorm_float32(
    device const float* input [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    constant uint& spatial [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_position_in_threadgroup]]
) {
    threadgroup float stats[2];
    uint channel = row % channels;
    uint rowOffset = row * spatial;

    if (threadIndex == 0) {
        float sum = 0.0f;

        for (uint offset = 0; offset < spatial; offset++) {
            sum += input[rowOffset + offset];
        }

        float mean = sum / float(spatial);
        float variance = 0.0f;

        for (uint offset = 0; offset < spatial; offset++) {
            float delta = input[rowOffset + offset] - mean;
            variance += delta * delta;
        }

        stats[0] = mean;
        stats[1] = 1.0f / precise::sqrt(variance / float(spatial) + layerNormEpsilonMetal);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean = stats[0];
    float invStdDev = stats[1];

    for (uint offset = threadIndex; offset < spatial; offset += normalizationThreadCount) {
        float centered = input[rowOffset + offset] - mean;
        out[rowOffset + offset] = fma(centered * invStdDev, scale[channel], bias[channel]);
    }
}

INSTANCENORM_KERNEL(instancenorm_float16, Float16NormStorage, half)
INSTANCENORM_KERNEL(instancenorm_bfloat16, BFloat16NormStorage, ushort)

BATCHNORM_EVAL_KERNEL(batchnorm_eval_float32, Float32NormStorage, float)
BATCHNORM_EVAL_KERNEL(batchnorm_eval_float16, Float16NormStorage, half)
BATCHNORM_EVAL_KERNEL(batchnorm_eval_bfloat16, BFloat16NormStorage, ushort)

kernel void inv_std_dev_float32(
    device const float* values [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) {
        return;
    }

    out[index] = 1.0f / precise::sqrt(values[index]);
}

kernel void groupnorm_stats_float32(
    device const float* input [[buffer(0)]],
    device float* meanOut [[buffer(1)]],
    device float* invStdDevOut [[buffer(2)]],
    constant uint& channels [[buffer(3)]],
    constant uint& spatial [[buffer(4)]],
    constant uint& groups [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_position_in_threadgroup]]
) {
    if (threadIndex != 0) {
        return;
    }

    uint groupIndex = row % groups;
    uint batchIndex = row / groups;
    uint channelsPerGroup = channels / groups;
    uint channelStart = groupIndex * channelsPerGroup;
    uint groupSize = channelsPerGroup * spatial;
    uint groupOffset = (batchIndex * channels + channelStart) * spatial;
    float sum = 0.0f;

    for (uint offset = 0; offset < groupSize; offset++) {
        sum += input[groupOffset + offset];
    }

    float mean = sum / float(groupSize);
    float variance = 0.0f;

    for (uint offset = 0; offset < groupSize; offset++) {
        float delta = input[groupOffset + offset] - mean;
        variance += delta * delta;
    }

    meanOut[row] = mean;
    invStdDevOut[row] = 1.0f / precise::sqrt(variance / float(groupSize) + layerNormEpsilonMetal);
}

kernel void instancenorm_stats_float32(
    device const float* input [[buffer(0)]],
    device float* meanOut [[buffer(1)]],
    device float* invStdDevOut [[buffer(2)]],
    constant uint& channels [[buffer(3)]],
    constant uint& spatial [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint threadIndex [[thread_position_in_threadgroup]]
) {
    if (threadIndex != 0) {
        return;
    }

    uint rowOffset = row * spatial;
    float sum = 0.0f;

    for (uint offset = 0; offset < spatial; offset++) {
        sum += input[rowOffset + offset];
    }

    float mean = sum / float(spatial);
    float variance = 0.0f;

    for (uint offset = 0; offset < spatial; offset++) {
        float delta = input[rowOffset + offset] - mean;
        variance += delta * delta;
    }

    meanOut[row] = mean;
    invStdDevOut[row] = 1.0f / precise::sqrt(variance / float(spatial) + layerNormEpsilonMetal);
}
