#include "elementwise.cuh"

extern "C" __global__ void axpy_float32(
    float* y,
    const float* x,
    unsigned int count,
    float alpha
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= count) {
        return;
    }

    y[index] += alpha * x[index];
}

extern "C" __global__ void axpy_float16(
    __half* y,
    const __half* x,
    unsigned int count,
    float alpha
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= count) {
        return;
    }

    float yValue = __half2float(y[index]);
    float xValue = __half2float(x[index]);
    y[index] = __float2half(yValue + alpha * xValue);
}

extern "C" __global__ void axpy_bfloat16(
    __nv_bfloat16* y,
    const __nv_bfloat16* x,
    unsigned int count,
    float alpha
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= count) {
        return;
    }

    float yValue = __bfloat162float(y[index]);
    float xValue = __bfloat162float(x[index]);
    y[index] = __float2bfloat16(yValue + alpha * xValue);
}
