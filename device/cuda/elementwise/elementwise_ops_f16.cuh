#ifndef PUTER_DEVICE_CUDA_ELEMENTWISE_ELEMENTWISE_OPS_F16_CUH
#define PUTER_DEVICE_CUDA_ELEMENTWISE_ELEMENTWISE_OPS_F16_CUH

static __device__ __forceinline__ __half elementwise_zero_h(void) {
    return __float2half(0.0f);
}

static __device__ __forceinline__ __half2 elementwise_abs_h2(__half2 value) {
    return __halves2half2(
        __habs(__low2half(value)),
        __habs(__high2half(value))
    );
}

static __device__ __forceinline__ __half elementwise_abs_h1(__half value) {
    return __habs(value);
}

static __device__ __forceinline__ __half2 elementwise_neg_h2(__half2 value) {
    return __hneg2(value);
}

static __device__ __forceinline__ __half elementwise_neg_h1(__half value) {
    return __hneg(value);
}

static __device__ __forceinline__ __half2 elementwise_sqrt_h2(__half2 value) {
    return __halves2half2(
        hsqrt(__low2half(value)),
        hsqrt(__high2half(value))
    );
}

static __device__ __forceinline__ __half elementwise_sqrt_h1(__half value) {
    return hsqrt(value);
}

static __device__ __forceinline__ __half2 elementwise_relu_h2(__half2 value) {
    __half zero = elementwise_zero_h();
    return __halves2half2(
        __hmax(__low2half(value), zero),
        __hmax(__high2half(value), zero)
    );
}

static __device__ __forceinline__ __half elementwise_relu_h1(__half value) {
    return __hmax(value, elementwise_zero_h());
}

static __device__ __forceinline__ __half2 elementwise_add_h2(__half2 left, __half2 right) {
    return __hadd2(left, right);
}

static __device__ __forceinline__ __half elementwise_add_h1(__half left, __half right) {
    return __hadd(left, right);
}

static __device__ __forceinline__ __half2 elementwise_sub_h2(__half2 left, __half2 right) {
    return __hsub2(left, right);
}

static __device__ __forceinline__ __half elementwise_sub_h1(__half left, __half right) {
    return __hsub(left, right);
}

static __device__ __forceinline__ __half2 elementwise_mul_h2(__half2 left, __half2 right) {
    return __hmul2(left, right);
}

static __device__ __forceinline__ __half elementwise_mul_h1(__half left, __half right) {
    return __hmul(left, right);
}

static __device__ __forceinline__ __half2 elementwise_div_h2(__half2 left, __half2 right) {
    return __h2div(left, right);
}

static __device__ __forceinline__ __half elementwise_div_h1(__half left, __half right) {
    return __hdiv(left, right);
}

static __device__ __forceinline__ __half2 elementwise_max_h2(__half2 left, __half2 right) {
    return __hmax2(left, right);
}

static __device__ __forceinline__ __half elementwise_max_h1(__half left, __half right) {
    return __hmax(left, right);
}

static __device__ __forceinline__ __half2 elementwise_min_h2(__half2 left, __half2 right) {
    return __hmin2(left, right);
}

static __device__ __forceinline__ __half elementwise_min_h1(__half left, __half right) {
    return __hmin(left, right);
}

#endif
