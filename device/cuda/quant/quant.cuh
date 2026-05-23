#ifndef PUTER_DEVICE_CUDA_QUANT_QUANT_CUH
#define PUTER_DEVICE_CUDA_QUANT_QUANT_CUH

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void int8_quant(
    const float* input,
    int8_t* out,
    unsigned int count
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= count) {
        return;
    }

    float rounded = roundf(input[index]);
    rounded = fminf(fmaxf(rounded, -128.0f), 127.0f);
    out[index] = static_cast<int8_t>(rounded);
}

#endif
