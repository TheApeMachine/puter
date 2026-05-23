#include "elementwise_unary_macros.cuh"
#include "elementwise_ops_f32.cuh"
#include "elementwise_ops_f16.cuh"
#include "elementwise_ops_bf16.cuh"

ELEMENTWISE_UNARY_KERNEL_F32(abs, elementwise_abs_f4, elementwise_abs_f1)
ELEMENTWISE_UNARY_KERNEL_F32(neg, elementwise_neg_f4, elementwise_neg_f1)
ELEMENTWISE_UNARY_KERNEL_F32(sqrt, elementwise_sqrt_f4, elementwise_sqrt_f1)
ELEMENTWISE_UNARY_KERNEL_F32(relu, elementwise_relu_f4, elementwise_relu_f1)

ELEMENTWISE_UNARY_KERNEL_F16(abs, elementwise_abs_h2, elementwise_abs_h1)
ELEMENTWISE_UNARY_KERNEL_F16(neg, elementwise_neg_h2, elementwise_neg_h1)
ELEMENTWISE_UNARY_KERNEL_F16(sqrt, elementwise_sqrt_h2, elementwise_sqrt_h1)
ELEMENTWISE_UNARY_KERNEL_F16(relu, elementwise_relu_h2, elementwise_relu_h1)

ELEMENTWISE_UNARY_KERNEL_BF16(abs, elementwise_abs_b2, elementwise_abs_bf16)
ELEMENTWISE_UNARY_KERNEL_BF16(neg, elementwise_neg_b2, elementwise_neg_bf16)
ELEMENTWISE_UNARY_KERNEL_BF16(sqrt, elementwise_sqrt_b2, elementwise_sqrt_bf16)
ELEMENTWISE_UNARY_KERNEL_BF16(relu, elementwise_relu_b2, elementwise_relu_bf16)
