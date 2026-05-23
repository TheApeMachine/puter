#ifndef PUTER_DEVICE_CUDA_ELEMENTWISE_ELEMENTWISE_OPS_BF16_CUH
#define PUTER_DEVICE_CUDA_ELEMENTWISE_ELEMENTWISE_OPS_BF16_CUH

static __device__ __forceinline__ __nv_bfloat16 elementwise_zero_bf16(void) {
    return __float2bfloat16(0.0f);
}

static __device__ __forceinline__ __nv_bfloat162 elementwise_abs_b2(__nv_bfloat162 value) {
    return __halves2bfloat162(
        __habs(__low2bfloat16(value)),
        __habs(__high2bfloat16(value))
    );
}

static __device__ __forceinline__ __nv_bfloat16 elementwise_abs_bf16(__nv_bfloat16 value) {
    return __habs(value);
}

static __device__ __forceinline__ __nv_bfloat162 elementwise_neg_b2(__nv_bfloat162 value) {
    return __hneg2(value);
}

static __device__ __forceinline__ __nv_bfloat16 elementwise_neg_bf16(__nv_bfloat16 value) {
    return __hneg(value);
}

static __device__ __forceinline__ __nv_bfloat162 elementwise_sqrt_b2(__nv_bfloat162 value) {
    return __halves2bfloat162(
        hsqrt(__low2bfloat16(value)),
        hsqrt(__high2bfloat16(value))
    );
}

static __device__ __forceinline__ __nv_bfloat16 elementwise_sqrt_bf16(__nv_bfloat16 value) {
    return hsqrt(value);
}

static __device__ __forceinline__ __nv_bfloat162 elementwise_relu_b2(__nv_bfloat162 value) {
    __nv_bfloat16 zero = elementwise_zero_bf16();
    return __halves2bfloat162(
        __hmax(__low2bfloat16(value), zero),
        __hmax(__high2bfloat16(value), zero)
    );
}

static __device__ __forceinline__ __nv_bfloat16 elementwise_relu_bf16(__nv_bfloat16 value) {
    return __hmax(value, elementwise_zero_bf16());
}

static __device__ __forceinline__ __nv_bfloat162 elementwise_add_b2(__nv_bfloat162 left, __nv_bfloat162 right) {
    return __hadd2(left, right);
}

static __device__ __forceinline__ __nv_bfloat16 elementwise_add_bf16(__nv_bfloat16 left, __nv_bfloat16 right) {
    return __hadd(left, right);
}

static __device__ __forceinline__ __nv_bfloat162 elementwise_sub_b2(__nv_bfloat162 left, __nv_bfloat162 right) {
    return __hsub2(left, right);
}

static __device__ __forceinline__ __nv_bfloat16 elementwise_sub_bf16(__nv_bfloat16 left, __nv_bfloat16 right) {
    return __hsub(left, right);
}

static __device__ __forceinline__ __nv_bfloat162 elementwise_mul_b2(__nv_bfloat162 left, __nv_bfloat162 right) {
    return __hmul2(left, right);
}

static __device__ __forceinline__ __nv_bfloat16 elementwise_mul_bf16(__nv_bfloat16 left, __nv_bfloat16 right) {
    return __hmul(left, right);
}

static __device__ __forceinline__ __nv_bfloat162 elementwise_div_b2(__nv_bfloat162 left, __nv_bfloat162 right) {
    return __h2div(left, right);
}

static __device__ __forceinline__ __nv_bfloat16 elementwise_div_bf16(__nv_bfloat16 left, __nv_bfloat16 right) {
    return __hdiv(left, right);
}

static __device__ __forceinline__ __nv_bfloat162 elementwise_max_b2(__nv_bfloat162 left, __nv_bfloat162 right) {
    return __hmax2(left, right);
}

static __device__ __forceinline__ __nv_bfloat16 elementwise_max_bf16(__nv_bfloat16 left, __nv_bfloat16 right) {
    return __hmax(left, right);
}

static __device__ __forceinline__ __nv_bfloat162 elementwise_min_b2(__nv_bfloat162 left, __nv_bfloat162 right) {
    return __hmin2(left, right);
}

static __device__ __forceinline__ __nv_bfloat16 elementwise_min_bf16(__nv_bfloat16 left, __nv_bfloat16 right) {
    return __hmin(left, right);
}

#endif
