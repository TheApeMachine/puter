#include <metal_stdlib>
#include "../internal/norm_stats.metalinc"

using namespace metal;

constant uint normalizationThreadCount = 256;

// Used by groupnorm / instancenorm / batchnorm kernels below. Clang's
// -Wunneeded-internal-declaration fires because it folds the constant
// into uses; the attribute silences the noise without changing codegen.
__attribute__((unused))
constant float layerNormEpsilonMetal = 1.0e-5f;

static inline float bf16_to_float_norm(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort float_to_bf16_norm(float value) {
    uint bits = as_type<uint>(value);

    if ((bits & 0x7fffffffu) > 0x7f800000u) {
        return ushort((bits >> 16) | 0x0040u);
    }

    uint rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    return ushort(rounded >> 16);
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

__attribute__((unused))
static inline float2 ds_add_pair(float2 left, float2 right) {
    float2 withHigh = ds_add_float(left, right.x);
    return ds_add_float(withHigh, right.y);
}

static inline float2 ds_neg(float2 value) {
    return float2(-value.x, -value.y);
}

__attribute__((unused))
static inline float2 ds_sub_from_float(float value, float2 subtrahend) {
    return ds_add_float(ds_neg(subtrahend), value);
}

__attribute__((unused))
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

__attribute__((unused))
static inline float2 ds_div_count(float2 value, uint count) {
    return ds_mul_float(value, 1.0f / float(count));
}

__attribute__((unused))
static inline float ds_to_float(float2 value) {
    return value.x + value.y;
}

__attribute__((unused))
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

static inline ushort float_to_f16_norm(float value) {
    uint bits = as_type<uint>(value);
    uint sign = bits & 0x80000000u;
    uint exponent = bits & 0x7F800000u;
    uint coefficient = bits & 0x007FFFFFu;

    if (exponent == 0x7F800000u) {
        uint nanBit = 0u;

        if (coefficient != 0u) {
            nanBit = 0x0200u;
        }

        return ushort((sign >> 16) | 0x7C00u | nanBit | (coefficient >> 13));
    }

    uint halfSign = sign >> 16;
    int unbiasedExponent = int(exponent >> 23) - 127;
    int halfExponent = unbiasedExponent + 15;

    if (halfExponent >= 0x1F) {
        return ushort(halfSign | 0x7C00u);
    }

    if (halfExponent <= 0) {
        if (14 - halfExponent > 24) {
            return ushort(halfSign);
        }

        uint widened = coefficient | 0x00800000u;
        uint halfCoefficient = widened >> uint(14 - halfExponent);
        uint roundBit = 1u << uint(13 - halfExponent);

        if ((widened & roundBit) != 0u && (widened & (3u * roundBit - 1u)) != 0u) {
            halfCoefficient++;
        }

        return ushort(halfSign | halfCoefficient);
    }

    uint halfExponentBits = uint(halfExponent) << 10;
    uint halfCoefficient = coefficient >> 13;
    uint roundBit = 0x00001000u;

    if ((coefficient & roundBit) != 0u && (coefficient & (3u * roundBit - 1u)) != 0u) {
        return ushort((halfSign | halfExponentBits | halfCoefficient) + 1u);
    }

    return ushort(halfSign | halfExponentBits | halfCoefficient);
}

static inline float f16_to_float_norm(half value) {
    ushort bits = as_type<ushort>(value);
    uint sign = uint(bits & 0x8000u) << 16;
    uint exp = uint(bits & 0x7C00u) >> 10;
    uint coef = uint(bits & 0x03FFu) << 13;

    if (exp == 0x1Fu) {
        if (coef == 0u) {
            return as_type<float>(sign | 0x7F800000u);
        }

        return as_type<float>(sign | 0x7FC00000u | coef);
    }

    if (exp == 0u) {
        if (coef == 0u) {
            return as_type<float>(sign);
        }

        exp = 1u;

        while ((coef & 0x7F800000u) == 0u) {
            coef <<= 1u;
            exp--;
        }

        coef &= 0x007FFFFFu;
    }

    return as_type<float>(sign | ((exp + 112u) << 23) | coef);
}

struct Float16NormStorage {
    static float load(device const half* values, uint index) {
        return f16_to_float_norm(values[index]);
    }

    static void store(device half* values, uint index, float value) {
        values[index] = as_type<half>(float_to_f16_norm(value));
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
    float channelStdDev = precise::sqrt(Storage::load(variance, channel) + layerNormEpsilonMetal);
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
    float sumF32 = reduce_sum<Storage, Scalar>(input, reduction, groupOffset, groupSize, threadIndex);
    ulong sum64 = metal_sf32_to64(as_type<uint>(sumF32));
    float mean = as_type<float>(metal_sf64_to32(metal_sf64_div(sum64, metal_sf64_int_to64(int(groupSize)))));
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

    float invStdDev = metal_norm_inv_std_dev_f32(reduction[0], groupSize);

    for (uint offset = threadIndex; offset < groupSize; offset += normalizationThreadCount) {
        uint channel = channelStart + offset / spatial;
        float inputValue = Storage::load(input, groupOffset + offset);
        float scaleValue = Storage::load(scale, channel);
        float biasValue = Storage::load(bias, channel);
        Storage::store(
            out,
            groupOffset + offset,
            (inputValue - mean) * invStdDev * scaleValue + biasValue
        );
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
    float mean = reduce_sum<Storage, Scalar>(input, reduction, rowOffset, spatial, threadIndex) /
        float(spatial);
    float localVariance = 0.0f;

    for (uint offset = threadIndex; offset < spatial; offset += normalizationThreadCount) {
        float delta = Storage::load(input, rowOffset + offset) - mean;
        localVariance += delta * delta;
    }

    reduction[threadIndex] = localVariance;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = normalizationThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            reduction[threadIndex] += reduction[threadIndex + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float invStdDev = 1.0f / sqrt(reduction[0] / float(spatial) + layerNormEpsilonMetal);

    for (uint offset = threadIndex; offset < spatial; offset += normalizationThreadCount) {
        float normalized = (Storage::load(input, rowOffset + offset) - mean) * invStdDev;
        float value = normalized * Storage::load(scale, channel) + Storage::load(bias, channel);
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
    float invStdDev = 1.0f / sqrt(Storage::load(variance, channel) + layerNormEpsilonMetal);
    float channelMean = Storage::load(mean, channel);
    float channelScale = Storage::load(scale, channel);
    float channelBias = Storage::load(bias, channel);

    for (uint offset = threadIndex; offset < spatial; offset += normalizationThreadCount) {
        float normalized = (Storage::load(input, rowOffset + offset) - channelMean) * invStdDev;
        Storage::store(out, rowOffset + offset, normalized * channelScale + channelBias);
    }
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
    threadgroup float reduction[normalizationThreadCount]; \
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
