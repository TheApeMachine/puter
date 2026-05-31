#include <metal_stdlib>

using namespace metal;

struct Optimizer4Config {
    float learningRate;
    float beta1;
    float beta2;
    float epsilon;
    float beta1Correction;
    float beta2Correction;
    float weightDecay;
};

struct Optimizer3Config {
    float learningRate;
    float epsilon;
    float decay;
    float beta1;
    float beta2;
    float momentum;
};

struct Optimizer2Config {
    float learningRate;
};

struct HebbianConfig {
    float learningRate;
    float decay;
};

struct LARSConfig {
    float learningRate;
    float momentum;
    float weightDecay;
    float trustCoeff;
    float epsilon;
};

static inline float optimizer_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

static inline ushort optimizer_float_to_bf16(float value) {
    return ushort(as_type<uint>(value) >> 16);
}

struct Float32OptimizerStorage {
    static float load(device const float* values, uint index) {
        return values[index];
    }

    static void store(device float* values, uint index, float value) {
        values[index] = value;
    }
};

struct Float16OptimizerStorage {
    static float load(device const half* values, uint index) {
        return float(values[index]);
    }

    static void store(device half* values, uint index, float value) {
        values[index] = half(value);
    }
};

struct BFloat16OptimizerStorage {
    static float load(device const ushort* values, uint index) {
        return optimizer_bf16_to_float(values[index]);
    }

    static void store(device ushort* values, uint index, float value) {
        values[index] = optimizer_float_to_bf16(value);
    }
};

static inline float optimizer_sign(float value) {
    if (value > 0.0f) {
        return 1.0f;
    }

    if (value < 0.0f) {
        return -1.0f;
    }

    return 0.0f;
}

template <typename Storage, typename Scalar>
static inline void adam_step_kernel(
    device const Scalar* params,
    device const Scalar* gradients,
    device float* first,
    device float* second,
    device Scalar* out,
    constant Optimizer4Config& config,
    uint count,
    uint index
) {
    if (index >= count) {
        return;
    }

    float param = Storage::load(params, index);
    float grad = Storage::load(gradients, index);
    float negBeta1 = 0.0f - config.beta1;
    float oneMinusBeta1 = 1.0f + negBeta1;
    float updatedFirst = config.beta1 * first[index];
    updatedFirst = updatedFirst + oneMinusBeta1 * grad;
    first[index] = updatedFirst;
    float gradSquared = grad * grad;
    float negBeta2 = 0.0f - config.beta2;
    float oneMinusBeta2 = 1.0f + negBeta2;
    float updatedSecond = config.beta2 * second[index];
    updatedSecond = updatedSecond + oneMinusBeta2 * gradSquared;
    second[index] = updatedSecond;
    float correctedFirst = updatedFirst / config.beta1Correction;
    float correctedSecond = updatedSecond / config.beta2Correction;
    float sqrtSecond = precise::sqrt(correctedSecond);
    float denominator = sqrtSecond + config.epsilon;
    float step = config.learningRate * correctedFirst / denominator;
    Storage::store(out, index, param - step);
}

template <typename Storage, typename Scalar>
static inline void adamw_step_kernel(
    device const Scalar* params,
    device const Scalar* gradients,
    device float* first,
    device float* second,
    device Scalar* out,
    constant Optimizer4Config& config,
    uint count,
    uint index
) {
    if (index >= count) {
        return;
    }

    float param = Storage::load(params, index);
    float grad = Storage::load(gradients, index);
    float negBeta1 = 0.0f - config.beta1;
    float oneMinusBeta1 = 1.0f + negBeta1;
    float updatedFirst = config.beta1 * first[index];
    updatedFirst = updatedFirst + oneMinusBeta1 * grad;
    first[index] = updatedFirst;
    float gradSquared = grad * grad;
    float negBeta2 = 0.0f - config.beta2;
    float oneMinusBeta2 = 1.0f + negBeta2;
    float updatedSecond = config.beta2 * second[index];
    updatedSecond = updatedSecond + oneMinusBeta2 * gradSquared;
    second[index] = updatedSecond;
    float correctedFirst = updatedFirst / config.beta1Correction;
    float correctedSecond = updatedSecond / config.beta2Correction;
    float sqrtSecond = precise::sqrt(correctedSecond);
    float denominator = sqrtSecond + config.epsilon;
    float gradStep = config.learningRate * correctedFirst / denominator;
    float decayStep = config.learningRate * config.weightDecay * param;
    Storage::store(out, index, param - gradStep - decayStep);
}

template <typename Storage, typename Scalar>
static inline void adamax_step_kernel(
    device const Scalar* params,
    device const Scalar* gradients,
    device float* first,
    device float* infinity,
    device Scalar* out,
    constant Optimizer4Config& config,
    uint count,
    uint index
) {
    if (index >= count) {
        return;
    }

    float param = Storage::load(params, index);
    float grad = Storage::load(gradients, index);
    first[index] = config.beta1 * first[index] + (1.0f - config.beta1) * grad;
    infinity[index] = max(config.beta2 * infinity[index], abs(grad));
    float correctedFirst = first[index] / config.beta1Correction;
    Storage::store(
        out, index,
        param - config.learningRate * correctedFirst / (infinity[index] + config.epsilon)
    );
}

template <typename Storage, typename Scalar>
static inline void optimizer3_step_kernel(
    device const Scalar* params,
    device const Scalar* gradients,
    device float* state,
    device Scalar* out,
    constant Optimizer3Config& config,
    uint operation,
    uint count,
    uint index
) {
    if (index >= count) {
        return;
    }

    float param = Storage::load(params, index);
    float grad = Storage::load(gradients, index);

    if (operation == 3u) {
        state[index] += grad * grad;
        Storage::store(
            out, index,
            param - config.learningRate * grad / (sqrt(state[index]) + config.epsilon)
        );
        return;
    }

    if (operation == 4u) {
        state[index] = config.decay * state[index] + (1.0f - config.decay) * grad * grad;
        Storage::store(
            out, index,
            param - config.learningRate * grad / (sqrt(state[index]) + config.epsilon)
        );
        return;
    }

    if (operation == 5u) {
        float update = config.beta1 * state[index] + (1.0f - config.beta1) * grad;
        Storage::store(out, index, param - config.learningRate * optimizer_sign(update));
        state[index] = config.beta2 * state[index] + (1.0f - config.beta2) * grad;
        return;
    }

    state[index] = config.momentum * state[index] + grad;
    Storage::store(out, index, param - config.learningRate * state[index]);
}

template <typename Storage, typename Scalar>
static inline void lbfgs_step_kernel(
    device const Scalar* params,
    device const Scalar* gradients,
    device Scalar* out,
    constant Optimizer2Config& config,
    uint count,
    uint index
) {
    if (index >= count) {
        return;
    }

    Storage::store(
        out, index,
        Storage::load(params, index) - config.learningRate * Storage::load(gradients, index)
    );
}

template <typename Storage, typename Scalar>
static inline void hebbian_step_kernel(
    device const Scalar* weights,
    device const Scalar* post,
    device const Scalar* pre,
    device Scalar* out,
    constant HebbianConfig& config,
    uint postCount,
    uint preCount,
    uint index
) {
    uint elementCount = postCount * preCount;

    if (index >= elementCount) {
        return;
    }

    uint postIndex = index / preCount;
    uint preIndex = index - postIndex * preCount;
    float updated = Storage::load(weights, index) * (1.0f - config.decay) +
        config.learningRate * Storage::load(post, postIndex) * Storage::load(pre, preIndex);
    Storage::store(out, index, updated);
}

template <typename Storage, typename Scalar>
static inline void lars_norms_kernel(
    device const Scalar* params,
    device const Scalar* gradients,
    device float* scratch,
    threadgroup float* paramSums,
    threadgroup float* gradSums,
    uint count,
    uint localIndex,
    uint groupIndex
) {
    uint index = groupIndex * 256u + localIndex;
    float param = 0.0f;
    float grad = 0.0f;

    if (index < count) {
        param = Storage::load(params, index);
        grad = Storage::load(gradients, index);
    }

    paramSums[localIndex] = param * param;
    gradSums[localIndex] = grad * grad;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (localIndex < stride) {
            paramSums[localIndex] += paramSums[localIndex + stride];
            gradSums[localIndex] += gradSums[localIndex + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (localIndex == 0u) {
        scratch[groupIndex * 2u] = paramSums[0];
        scratch[groupIndex * 2u + 1u] = gradSums[0];
    }
}

template <typename Storage, typename Scalar>
static inline void lars_step_kernel(
    device const Scalar* params,
    device const Scalar* gradients,
    device float* momentum,
    device const float* scratch,
    device Scalar* out,
    constant LARSConfig& config,
    uint count,
    uint groupCount,
    uint index
) {
    if (index >= count) {
        return;
    }

    float paramSum = 0.0f;
    float gradSum = 0.0f;

    for (uint groupIndex = 0; groupIndex < groupCount; groupIndex++) {
        paramSum += scratch[groupIndex * 2u];
        gradSum += scratch[groupIndex * 2u + 1u];
    }

    float paramNorm = sqrt(paramSum);
    float gradNorm = sqrt(gradSum);
    float trust = 1.0f;

    if (paramNorm > 0.0f && gradNorm > 0.0f) {
        trust = config.trustCoeff * paramNorm /
            (gradNorm + config.weightDecay * paramNorm + config.epsilon);
    }

    float param = Storage::load(params, index);
    float decayed = Storage::load(gradients, index) + config.weightDecay * param;
    momentum[index] = config.momentum * momentum[index] + decayed;
    Storage::store(out, index, param - config.learningRate * trust * momentum[index]);
}

#define OPTIMIZER4_KERNEL(name, storage, scalar, body) \
kernel void name( \
    device const scalar* params [[buffer(0)]], \
    device const scalar* gradients [[buffer(1)]], \
    device float* first [[buffer(2)]], \
    device float* second [[buffer(3)]], \
    device scalar* out [[buffer(4)]], \
    constant uint& count [[buffer(5)]], \
    constant Optimizer4Config& config [[buffer(6)]], \
    uint index [[thread_position_in_grid]] \
) { \
    body<storage, scalar>(params, gradients, first, second, out, config, count, index); \
}

#define OPTIMIZER3_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* params [[buffer(0)]], \
    device const scalar* gradients [[buffer(1)]], \
    device float* state [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    constant uint& count [[buffer(4)]], \
    constant uint& operation [[buffer(5)]], \
    constant Optimizer3Config& config [[buffer(6)]], \
    uint index [[thread_position_in_grid]] \
) { \
    optimizer3_step_kernel<storage, scalar>(params, gradients, state, out, config, operation, count, index); \
}

#define OPTIMIZER2_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* params [[buffer(0)]], \
    device const scalar* gradients [[buffer(1)]], \
    device scalar* out [[buffer(2)]], \
    constant uint& count [[buffer(3)]], \
    constant Optimizer2Config& config [[buffer(4)]], \
    uint index [[thread_position_in_grid]] \
) { \
    lbfgs_step_kernel<storage, scalar>(params, gradients, out, config, count, index); \
}

#define HEBBIAN_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* weights [[buffer(0)]], \
    device const scalar* post [[buffer(1)]], \
    device const scalar* pre [[buffer(2)]], \
    device scalar* out [[buffer(3)]], \
    constant uint& postCount [[buffer(4)]], \
    constant uint& preCount [[buffer(5)]], \
    constant HebbianConfig& config [[buffer(6)]], \
    uint index [[thread_position_in_grid]] \
) { \
    hebbian_step_kernel<storage, scalar>(weights, post, pre, out, config, postCount, preCount, index); \
}

#define LARS_NORMS_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* params [[buffer(0)]], \
    device const scalar* gradients [[buffer(1)]], \
    device float* scratch [[buffer(2)]], \
    constant uint& count [[buffer(3)]], \
    uint localIndex [[thread_position_in_threadgroup]], \
    uint groupIndex [[threadgroup_position_in_grid]] \
) { \
    threadgroup float paramSums[256]; \
    threadgroup float gradSums[256]; \
    lars_norms_kernel<storage, scalar>( \
        params, gradients, scratch, paramSums, gradSums, count, localIndex, groupIndex \
    ); \
}

#define LARS_STEP_KERNEL(name, storage, scalar) \
kernel void name( \
    device const scalar* params [[buffer(0)]], \
    device const scalar* gradients [[buffer(1)]], \
    device float* momentum [[buffer(2)]], \
    device const float* scratch [[buffer(3)]], \
    device scalar* out [[buffer(4)]], \
    constant uint& count [[buffer(5)]], \
    constant uint& groupCount [[buffer(6)]], \
    constant LARSConfig& config [[buffer(7)]], \
    uint index [[thread_position_in_grid]] \
) { \
    lars_step_kernel<storage, scalar>(params, gradients, momentum, scratch, out, config, count, groupCount, index); \
}

OPTIMIZER4_KERNEL(adam_step_float32, Float32OptimizerStorage, float, adam_step_kernel)
OPTIMIZER4_KERNEL(adam_step_float16, Float16OptimizerStorage, half, adam_step_kernel)
OPTIMIZER4_KERNEL(adam_step_bfloat16, BFloat16OptimizerStorage, ushort, adam_step_kernel)
OPTIMIZER4_KERNEL(adamw_step_float32, Float32OptimizerStorage, float, adamw_step_kernel)
OPTIMIZER4_KERNEL(adamw_step_float16, Float16OptimizerStorage, half, adamw_step_kernel)
OPTIMIZER4_KERNEL(adamw_step_bfloat16, BFloat16OptimizerStorage, ushort, adamw_step_kernel)
OPTIMIZER4_KERNEL(adamax_step_float32, Float32OptimizerStorage, float, adamax_step_kernel)
OPTIMIZER4_KERNEL(adamax_step_float16, Float16OptimizerStorage, half, adamax_step_kernel)
OPTIMIZER4_KERNEL(adamax_step_bfloat16, BFloat16OptimizerStorage, ushort, adamax_step_kernel)

OPTIMIZER3_KERNEL(optimizer3_float32, Float32OptimizerStorage, float)
OPTIMIZER3_KERNEL(optimizer3_float16, Float16OptimizerStorage, half)
OPTIMIZER3_KERNEL(optimizer3_bfloat16, BFloat16OptimizerStorage, ushort)
OPTIMIZER2_KERNEL(lbfgs_step_float32, Float32OptimizerStorage, float)
OPTIMIZER2_KERNEL(lbfgs_step_float16, Float16OptimizerStorage, half)
OPTIMIZER2_KERNEL(lbfgs_step_bfloat16, BFloat16OptimizerStorage, ushort)
HEBBIAN_KERNEL(hebbian_step_float32, Float32OptimizerStorage, float)
HEBBIAN_KERNEL(hebbian_step_float16, Float16OptimizerStorage, half)
HEBBIAN_KERNEL(hebbian_step_bfloat16, BFloat16OptimizerStorage, ushort)
LARS_NORMS_KERNEL(lars_norms_float32, Float32OptimizerStorage, float)
LARS_NORMS_KERNEL(lars_norms_float16, Float16OptimizerStorage, half)
LARS_NORMS_KERNEL(lars_norms_bfloat16, BFloat16OptimizerStorage, ushort)
LARS_STEP_KERNEL(lars_step_float32, Float32OptimizerStorage, float)
LARS_STEP_KERNEL(lars_step_float16, Float16OptimizerStorage, half)
LARS_STEP_KERNEL(lars_step_bfloat16, BFloat16OptimizerStorage, ushort)
