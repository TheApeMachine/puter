#include <metal_stdlib>

using namespace metal;

constant uint lossThreadCount = 256;

static inline float loss_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort loss_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

struct Float32LossStorage {
    static float load(device const float* values, uint index) {
        return values[index];
    }

    static void store(device float* values, uint index, float value) {
        values[index] = value;
    }
};

struct Float16LossStorage {
    static float load(device const half* values, uint index) {
        return float(values[index]);
    }

    static void store(device half* values, uint index, float value) {
        values[index] = half(value);
    }
};

struct BFloat16LossStorage {
    static float load(device const ushort* values, uint index) {
        return loss_bf16_to_float(values[index]);
    }

    static void store(device ushort* values, uint index, float value) {
        values[index] = loss_float_to_bf16(value);
    }
};

static inline void set_loss_error(device atomic_uint* errorFlag) {
    atomic_store_explicit(errorFlag, 1u, memory_order_relaxed);
}

static inline float loss_safe_probability(float value) {
    return clamp(value, 1.0e-7f, 1.0f - 1.0e-7f);
}

static inline float loss_safe_positive(float value) {
    return max(value, 1.0e-12f);
}

struct MSELossOp {
    float operator()(float prediction, float target) const {
        float delta = prediction - target;
        return delta * delta;
    }
};

struct MAELossOp {
    float operator()(float prediction, float target) const {
        return abs(prediction - target);
    }
};

struct HuberLossOp {
    float operator()(float prediction, float target) const {
        float delta = prediction - target;
        float magnitude = abs(delta);

        if (magnitude <= 1.0f) {
            return 0.5f * delta * delta;
        }

        return magnitude - 0.5f;
    }
};

struct BinaryCrossEntropyLossOp {
    float operator()(float prediction, float target) const {
        float safePrediction = loss_safe_probability(prediction);
        return -target * log(safePrediction) -
            (1.0f - target) * log(1.0f - safePrediction);
    }
};

struct KLDivergenceLossOp {
    float operator()(float prediction, float target) const {
        float safePrediction = loss_safe_positive(prediction);
        float safeTarget = loss_safe_positive(target);
        return safeTarget * log(safeTarget / safePrediction);
    }
};

template <typename Storage, typename Scalar, typename LossOp>
static inline void pair_loss_partial(
    device const Scalar* predictions,
    device const Scalar* targets,
    device float* scratch,
    threadgroup float* reduction,
    constant uint& count,
    uint groupIndex,
    uint threadIndex,
    LossOp op
) {
    uint valueIndex = groupIndex * lossThreadCount + threadIndex;
    float localValue = 0.0f;

    if (valueIndex < count) {
        localValue = op(
            Storage::load(predictions, valueIndex),
            Storage::load(targets, valueIndex)
        );
    }

    reduction[threadIndex] = localValue;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = lossThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            reduction[threadIndex] += reduction[threadIndex + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (threadIndex == 0) {
        scratch[groupIndex] = reduction[0];
    }
}

template <typename Storage, typename Scalar>
static inline void loss_finalize(
    device const float* scratch,
    device Scalar* out,
    threadgroup float* reduction,
    constant uint& partialCount,
    constant uint& denominator,
    uint threadIndex
) {
    float localValue = 0.0f;

    for (uint index = threadIndex; index < partialCount; index += lossThreadCount) {
        localValue += scratch[index];
    }

    reduction[threadIndex] = localValue;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = lossThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            reduction[threadIndex] += reduction[threadIndex + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (threadIndex == 0) {
        Storage::store(out, 0, reduction[0] / float(denominator));
    }
}

template <typename Storage, typename Scalar>
static inline void cross_entropy_loss_partial(
    device const Scalar* logits,
    device const int* targets,
    device float* scratch,
    device atomic_uint* errorFlag,
    threadgroup float* reduction,
    constant uint& classes,
    uint rowIndex,
    uint threadIndex
) {
    int targetID = targets[rowIndex];
    bool targetOK = targetID >= 0 && uint(targetID) < classes;
    uint rowOffset = rowIndex * classes;
    float localMax = -3.4028234663852886e38f;

    if (!targetOK && threadIndex == 0) {
        set_loss_error(errorFlag);
    }

    for (uint col = threadIndex; targetOK && col < classes; col += lossThreadCount) {
        localMax = max(localMax, Storage::load(logits, rowOffset + col));
    }

    reduction[threadIndex] = localMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = lossThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            reduction[threadIndex] = max(reduction[threadIndex], reduction[threadIndex + stride]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float maximum = reduction[0];
    float localSum = 0.0f;

    for (uint col = threadIndex; targetOK && col < classes; col += lossThreadCount) {
        localSum += exp(Storage::load(logits, rowOffset + col) - maximum);
    }

    reduction[threadIndex] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = lossThreadCount / 2; stride > 0; stride >>= 1) {
        if (threadIndex < stride) {
            reduction[threadIndex] += reduction[threadIndex + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (threadIndex == 0) {
        float targetLogit = targetOK ? Storage::load(logits, rowOffset + uint(targetID)) : 0.0f;
        scratch[rowIndex] = targetOK ? -(targetLogit - maximum - log(reduction[0])) : 0.0f;
    }
}
