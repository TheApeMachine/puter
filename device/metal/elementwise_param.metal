#include <metal_stdlib>

using namespace metal;

template <typename UnaryOp>
static inline void param_unary_float32(
    device const float4* inputVector,
    device float4* outVector,
    constant uint& count,
    constant float& param,
    uint index,
    UnaryOp op
) {
    uint base = index * 4;

    if (base + 3 < count) {
        float4 value = float4(inputVector[index]);
        outVector[index] = op(value, param);
        return;
    }

    device const float* input = reinterpret_cast<device const float*>(inputVector);
    device float* out = reinterpret_cast<device float*>(outVector);

    for (uint offset = 0; offset < 4; offset++) {
        uint scalarIndex = base + offset;

        if (scalarIndex < count) {
            out[scalarIndex] = op(input[scalarIndex], param);
        }
    }
}

struct PReLUSlopeOp {
    float4 operator()(float4 value, float slope) const {
        return select(slope * value, value, value > float4(0.0f));
    }
    float operator()(float value, float slope) const {
        return value > 0.0f ? value : slope * value;
    }
};

struct LeakyReLUSlopeOp {
    float4 operator()(float4 value, float slope) const {
        return select(slope * value, value, value > float4(0.0f));
    }
    float operator()(float value, float slope) const {
        return value > 0.0f ? value : slope * value;
    }
};

struct ELUAlphaOp {
    float4 operator()(float4 value, float alpha) const {
        return select(alpha * (precise::exp(value) - float4(1.0f)), value, value > float4(0.0f));
    }
    float operator()(float value, float alpha) const {
        return value > 0.0f ? value : alpha * (precise::exp(value) - 1.0f);
    }
};

struct CELUAlphaOp {
    float4 operator()(float4 value, float alpha) const {
        return select(alpha * (precise::exp(value / alpha) - float4(1.0f)), value, value > float4(0.0f));
    }
    float operator()(float value, float alpha) const {
        return value > 0.0f ? value : alpha * (precise::exp(value / alpha) - 1.0f);
    }
};

struct ThresholdOp {
    float4 operator()(float4 value, float threshold) const {
        return select(float4(0.0f), value, value > float4(threshold));
    }
    float operator()(float value, float threshold) const {
        return value > threshold ? value : 0.0f;
    }
};

#define PARAM_UNARY_KERNEL(name, op) \
kernel void name##_float32( \
    device const float4* inputVector [[buffer(0)]], \
    device float4* outVector [[buffer(1)]], \
    constant uint& count [[buffer(2)]], \
    constant float& param [[buffer(3)]], \
    uint index [[thread_position_in_grid]] \
) { \
    param_unary_float32(inputVector, outVector, count, param, index, op{}); \
}

PARAM_UNARY_KERNEL(prelu_slope, PReLUSlopeOp)
PARAM_UNARY_KERNEL(leaky_relu_slope, LeakyReLUSlopeOp)
PARAM_UNARY_KERNEL(elu_alpha, ELUAlphaOp)
PARAM_UNARY_KERNEL(celu_alpha, CELUAlphaOp)
PARAM_UNARY_KERNEL(threshold, ThresholdOp)
