#include "hawkes.cuh"

extern "C" __global__ void hawkes_log_likelihood_float32_partial(
    const float* events,
    const float* totalTime,
    const float* baseline,
    const float* alpha,
    const float* beta,
    float* scratch,
    unsigned int eventCount
) {
    __shared__ float reduction[cudaHawkesMarkovThreadCount];
    hawkes_log_likelihood_partial_kernel<Float32HawkesMarkovStorage, float>(
        events,
        totalTime,
        baseline,
        alpha,
        beta,
        scratch,
        reduction,
        eventCount,
        blockIdx.x,
        threadIdx.x
    );
}

extern "C" __global__ void hawkes_log_likelihood_float32_finalize(
    const float* scratch,
    const float* totalTime,
    const float* baseline,
    float* out,
    unsigned int eventCount
) {
    __shared__ float reduction[cudaHawkesMarkovThreadCount];
    hawkes_log_likelihood_finalize_kernel<Float32HawkesMarkovStorage, float>(
        scratch,
        totalTime,
        baseline,
        out,
        reduction,
        eventCount,
        threadIdx.x
    );
}

extern "C" __global__ void hawkes_log_likelihood_float16_partial(
    const __half* events,
    const __half* totalTime,
    const __half* baseline,
    const __half* alpha,
    const __half* beta,
    float* scratch,
    unsigned int eventCount
) {
    __shared__ float reduction[cudaHawkesMarkovThreadCount];
    hawkes_log_likelihood_partial_kernel<Float16HawkesMarkovStorage, __half>(
        events,
        totalTime,
        baseline,
        alpha,
        beta,
        scratch,
        reduction,
        eventCount,
        blockIdx.x,
        threadIdx.x
    );
}

extern "C" __global__ void hawkes_log_likelihood_float16_finalize(
    const float* scratch,
    const __half* totalTime,
    const __half* baseline,
    __half* out,
    unsigned int eventCount
) {
    __shared__ float reduction[cudaHawkesMarkovThreadCount];
    hawkes_log_likelihood_finalize_kernel<Float16HawkesMarkovStorage, __half>(
        scratch,
        totalTime,
        baseline,
        out,
        reduction,
        eventCount,
        threadIdx.x
    );
}

extern "C" __global__ void hawkes_log_likelihood_bfloat16_partial(
    const unsigned short* events,
    const unsigned short* totalTime,
    const unsigned short* baseline,
    const unsigned short* alpha,
    const unsigned short* beta,
    float* scratch,
    unsigned int eventCount
) {
    __shared__ float reduction[cudaHawkesMarkovThreadCount];
    hawkes_log_likelihood_partial_kernel<BFloat16HawkesMarkovStorage, unsigned short>(
        events,
        totalTime,
        baseline,
        alpha,
        beta,
        scratch,
        reduction,
        eventCount,
        blockIdx.x,
        threadIdx.x
    );
}

extern "C" __global__ void hawkes_log_likelihood_bfloat16_finalize(
    const float* scratch,
    const unsigned short* totalTime,
    const unsigned short* baseline,
    unsigned short* out,
    unsigned int eventCount
) {
    __shared__ float reduction[cudaHawkesMarkovThreadCount];
    hawkes_log_likelihood_finalize_kernel<BFloat16HawkesMarkovStorage, unsigned short>(
        scratch,
        totalTime,
        baseline,
        out,
        reduction,
        eventCount,
        threadIdx.x
    );
}
