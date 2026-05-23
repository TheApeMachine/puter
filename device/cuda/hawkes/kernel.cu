#include "hawkes.cuh"

extern "C" __global__ void hawkes_kernel_matrix_float32(
    const float* events,
    const float* alpha,
    const float* beta,
    float* out,
    unsigned int eventCount
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    hawkes_kernel_matrix_kernel<Float32HawkesMarkovStorage, float>(
        events,
        alpha,
        beta,
        out,
        eventCount,
        index
    );
}

extern "C" __global__ void hawkes_kernel_matrix_float16(
    const __half* events,
    const __half* alpha,
    const __half* beta,
    __half* out,
    unsigned int eventCount
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    hawkes_kernel_matrix_kernel<Float16HawkesMarkovStorage, __half>(
        events,
        alpha,
        beta,
        out,
        eventCount,
        index
    );
}

extern "C" __global__ void hawkes_kernel_matrix_bfloat16(
    const unsigned short* events,
    const unsigned short* alpha,
    const unsigned short* beta,
    unsigned short* out,
    unsigned int eventCount
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    hawkes_kernel_matrix_kernel<BFloat16HawkesMarkovStorage, unsigned short>(
        events,
        alpha,
        beta,
        out,
        eventCount,
        index
    );
}

extern "C" __global__ void hawkes_exp_float32(
    const float* inputVector,
    float* outVector,
    unsigned int count
) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= count) {
        return;
    }

    outVector[index] = metal_hawkes_exp32(inputVector[index]);
}
