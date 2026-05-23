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

extern "C" __global__ void int8_dequant(
    float* destination,
    const int8_t* source,
    float scale,
    int zeroPoint,
    unsigned int count
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= count) {
        return;
    }

    destination[index] = static_cast<float>(static_cast<int>(source[index]) - zeroPoint) * scale;
}

extern "C" __global__ void int4_dequant(
    float* destination,
    const int8_t* source,
    float scale,
    int zeroPoint,
    unsigned int pairCount
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= pairCount) {
        return;
    }

    unsigned int byteIndex = index / 2u;
    unsigned int nibble = index & 1u;
    int packed = static_cast<int>(source[byteIndex]);
    int value = (nibble == 0u) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);

    if (value >= 8) {
        value -= 16;
    }

    destination[index] = static_cast<float>(value - zeroPoint) * scale;
}

#endif
