#include "hawkes.cuh"

extern "C" __global__ void hawkes_intensity_float32(
    const float* events,
    const float* queryTimes,
    const float* baseline,
    const float* alpha,
    const float* beta,
    float* out,
    unsigned int eventCount
) {
    __shared__ float reduction[cudaHawkesMarkovThreadCount];
    hawkes_intensity_kernel<Float32HawkesMarkovStorage, float>(
        events,
        queryTimes,
        baseline,
        alpha,
        beta,
        out,
        reduction,
        eventCount,
        blockIdx.x,
        threadIdx.x
    );
}

extern "C" __global__ void hawkes_intensity_float16(
    const __half* events,
    const __half* queryTimes,
    const __half* baseline,
    const __half* alpha,
    const __half* beta,
    __half* out,
    unsigned int eventCount
) {
    __shared__ float reduction[cudaHawkesMarkovThreadCount];
    hawkes_intensity_kernel<Float16HawkesMarkovStorage, __half>(
        events,
        queryTimes,
        baseline,
        alpha,
        beta,
        out,
        reduction,
        eventCount,
        blockIdx.x,
        threadIdx.x
    );
}

extern "C" __global__ void hawkes_intensity_bfloat16(
    const unsigned short* events,
    const unsigned short* queryTimes,
    const unsigned short* baseline,
    const unsigned short* alpha,
    const unsigned short* beta,
    unsigned short* out,
    unsigned int eventCount
) {
    __shared__ float reduction[cudaHawkesMarkovThreadCount];
    hawkes_intensity_kernel<BFloat16HawkesMarkovStorage, unsigned short>(
        events,
        queryTimes,
        baseline,
        alpha,
        beta,
        out,
        reduction,
        eventCount,
        blockIdx.x,
        threadIdx.x
    );
}
