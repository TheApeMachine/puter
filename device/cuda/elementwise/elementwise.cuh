#ifndef PUTER_DEVICE_CUDA_ELEMENTWISE_ELEMENTWISE_CUH
#define PUTER_DEVICE_CUDA_ELEMENTWISE_ELEMENTWISE_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>

__device__ __forceinline__ float elementwise_bf16_to_float(unsigned short value) {
    unsigned int bits = static_cast<unsigned int>(value) << 16U;
    return __uint_as_float(bits);
}

__device__ __forceinline__ unsigned short elementwise_float_to_bf16(float value) {
    return static_cast<unsigned short>(__float_as_uint(value) >> 16U);
}

#endif
