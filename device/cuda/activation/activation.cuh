#ifndef PUTER_DEVICE_CUDA_ACTIVATION_ACTIVATION_CUH
#define PUTER_DEVICE_CUDA_ACTIVATION_ACTIVATION_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

static __device__ __forceinline__ float activation_bf16_to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

static __device__ __forceinline__ __nv_bfloat16 activation_float_to_bf16(float value) {
    return __float2bfloat16(value);
}

static __device__ __forceinline__ float activation_selu_scale(void) {
    return 1.05070098735548049342f;
}

static __device__ __forceinline__ float activation_selu_alpha(void) {
    return 1.67326324235437728482f;
}

static __device__ __forceinline__ float activation_leaky_relu_slope(void) {
    return 0.01f;
}

#endif
