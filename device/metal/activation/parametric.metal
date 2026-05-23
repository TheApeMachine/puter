#include <metal_stdlib>
#include "activation_parametric_ops_f32.metalinc"
#include "activation_parametric_ops_f16.metalinc"
#include "activation_parametric_ops_bf16.metalinc"
#include "activation_parametric.macros.metalinc"

using namespace metal;

PARAM_UNARY_KERNEL_F32(prelu_slope, PReLUSlopeOp)
PARAM_UNARY_KERNEL_F16(prelu_slope, HalfPReLUSlopeOp)
PARAM_UNARY_KERNEL_BF16(prelu_slope, BF16PReLUSlopeOp)

PARAM_UNARY_KERNEL_F32(leaky_relu_slope, LeakyReLUSlopeOp)
PARAM_UNARY_KERNEL_F16(leaky_relu_slope, HalfLeakyReLUSlopeOp)
PARAM_UNARY_KERNEL_BF16(leaky_relu_slope, BF16LeakyReLUSlopeOp)

PARAM_UNARY_KERNEL_F32(elu_alpha, ELUAlphaOp)
PARAM_UNARY_KERNEL_F16(elu_alpha, HalfELUAlphaOp)
PARAM_UNARY_KERNEL_BF16(elu_alpha, BF16ELUAlphaOp)

PARAM_UNARY_KERNEL_F32(celu_alpha, CELUAlphaOp)
PARAM_UNARY_KERNEL_F16(celu_alpha, HalfCELUAlphaOp)
PARAM_UNARY_KERNEL_BF16(celu_alpha, BF16CELUAlphaOp)

PARAM_UNARY_KERNEL_F32(threshold, ThresholdOp)
PARAM_UNARY_KERNEL_F16(threshold, HalfThresholdOp)
PARAM_UNARY_KERNEL_BF16(threshold, BF16ThresholdOp)
