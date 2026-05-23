#include "elementwise_binary_macros.cuh"
#include "elementwise_ops_f32.cuh"
#include "elementwise_ops_f16.cuh"
#include "elementwise_ops_bf16.cuh"

ELEMENTWISE_BINARY_KERNEL_F32(add, elementwise_add_f4, elementwise_add_f1)
ELEMENTWISE_BINARY_KERNEL_F32(sub, elementwise_sub_f4, elementwise_sub_f1)
ELEMENTWISE_BINARY_KERNEL_F32(mul, elementwise_mul_f4, elementwise_mul_f1)
ELEMENTWISE_BINARY_KERNEL_F32(div, elementwise_div_f4, elementwise_div_f1)
ELEMENTWISE_BINARY_KERNEL_F32(max, elementwise_max_f4, elementwise_max_f1)
ELEMENTWISE_BINARY_KERNEL_F32(min, elementwise_min_f4, elementwise_min_f1)

ELEMENTWISE_BINARY_KERNEL_F16(add, elementwise_add_h2, elementwise_add_h1)
ELEMENTWISE_BINARY_KERNEL_F16(sub, elementwise_sub_h2, elementwise_sub_h1)
ELEMENTWISE_BINARY_KERNEL_F16(mul, elementwise_mul_h2, elementwise_mul_h1)
ELEMENTWISE_BINARY_KERNEL_F16(div, elementwise_div_h2, elementwise_div_h1)
ELEMENTWISE_BINARY_KERNEL_F16(max, elementwise_max_h2, elementwise_max_h1)
ELEMENTWISE_BINARY_KERNEL_F16(min, elementwise_min_h2, elementwise_min_h1)

ELEMENTWISE_BINARY_KERNEL_BF16(add, elementwise_add_b2, elementwise_add_bf16)
ELEMENTWISE_BINARY_KERNEL_BF16(sub, elementwise_sub_b2, elementwise_sub_bf16)
ELEMENTWISE_BINARY_KERNEL_BF16(mul, elementwise_mul_b2, elementwise_mul_bf16)
ELEMENTWISE_BINARY_KERNEL_BF16(div, elementwise_div_b2, elementwise_div_bf16)
ELEMENTWISE_BINARY_KERNEL_BF16(max, elementwise_max_b2, elementwise_max_bf16)
ELEMENTWISE_BINARY_KERNEL_BF16(min, elementwise_min_b2, elementwise_min_bf16)
