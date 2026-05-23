#include "hawkes.cuh"

extern "C" __global__ void markov_mutual_information_float32_partial(
    const float* joint,
    float* scratch,
    unsigned int rows,
    unsigned int cols
) {
    __shared__ float reduction[cudaHawkesMarkovThreadCount];
    markov_mutual_information_partial_kernel<Float32HawkesMarkovStorage, float>(
        joint,
        scratch,
        reduction,
        rows,
        cols,
        blockIdx.x,
        threadIdx.x
    );
}

extern "C" __global__ void hawkes_markov_finalize_float32(
    const float* scratch,
    float* out,
    unsigned int partialCount
) {
    __shared__ float reduction[cudaHawkesMarkovThreadCount];
    hawkes_markov_finalize_kernel<Float32HawkesMarkovStorage, float>(
        scratch,
        out,
        reduction,
        partialCount,
        threadIdx.x
    );
}

extern "C" __global__ void markov_blanket_partition_float32(
    const float* adjacency,
    const int* internalNodes,
    int* out,
    unsigned int nodeCount,
    unsigned int internalCount
) {
    unsigned int nodeIndex = blockIdx.x * blockDim.x + threadIdx.x;
    markov_blanket_partition_kernel<Float32HawkesMarkovStorage, float>(
        adjacency,
        internalNodes,
        out,
        nodeCount,
        internalCount,
        nodeIndex
    );
}

extern "C" __global__ void markov_flow_float32(
    const float* mutualInformation,
    const int* partition,
    float* out,
    unsigned int nodeCount,
    int targetLabel
) {
    unsigned int nodeIndex = blockIdx.x * blockDim.x + threadIdx.x;
    markov_flow_kernel<Float32HawkesMarkovStorage, float>(
        mutualInformation,
        partition,
        out,
        nodeCount,
        targetLabel,
        nodeIndex
    );
}

extern "C" __global__ void markov_mutual_information_float16_partial(
    const __half* joint,
    float* scratch,
    unsigned int rows,
    unsigned int cols
) {
    __shared__ float reduction[cudaHawkesMarkovThreadCount];
    markov_mutual_information_partial_kernel<Float16HawkesMarkovStorage, __half>(
        joint,
        scratch,
        reduction,
        rows,
        cols,
        blockIdx.x,
        threadIdx.x
    );
}

extern "C" __global__ void hawkes_markov_finalize_float16(
    const float* scratch,
    __half* out,
    unsigned int partialCount
) {
    __shared__ float reduction[cudaHawkesMarkovThreadCount];
    hawkes_markov_finalize_kernel<Float16HawkesMarkovStorage, __half>(
        scratch,
        out,
        reduction,
        partialCount,
        threadIdx.x
    );
}

extern "C" __global__ void markov_blanket_partition_float16(
    const __half* adjacency,
    const int* internalNodes,
    int* out,
    unsigned int nodeCount,
    unsigned int internalCount
) {
    unsigned int nodeIndex = blockIdx.x * blockDim.x + threadIdx.x;
    markov_blanket_partition_kernel<Float16HawkesMarkovStorage, __half>(
        adjacency,
        internalNodes,
        out,
        nodeCount,
        internalCount,
        nodeIndex
    );
}

extern "C" __global__ void markov_flow_float16(
    const __half* mutualInformation,
    const int* partition,
    __half* out,
    unsigned int nodeCount,
    int targetLabel
) {
    unsigned int nodeIndex = blockIdx.x * blockDim.x + threadIdx.x;
    markov_flow_kernel<Float16HawkesMarkovStorage, __half>(
        mutualInformation,
        partition,
        out,
        nodeCount,
        targetLabel,
        nodeIndex
    );
}

extern "C" __global__ void markov_mutual_information_bfloat16_partial(
    const unsigned short* joint,
    float* scratch,
    unsigned int rows,
    unsigned int cols
) {
    __shared__ float reduction[cudaHawkesMarkovThreadCount];
    markov_mutual_information_partial_kernel<BFloat16HawkesMarkovStorage, unsigned short>(
        joint,
        scratch,
        reduction,
        rows,
        cols,
        blockIdx.x,
        threadIdx.x
    );
}

extern "C" __global__ void hawkes_markov_finalize_bfloat16(
    const float* scratch,
    unsigned short* out,
    unsigned int partialCount
) {
    __shared__ float reduction[cudaHawkesMarkovThreadCount];
    hawkes_markov_finalize_kernel<BFloat16HawkesMarkovStorage, unsigned short>(
        scratch,
        out,
        reduction,
        partialCount,
        threadIdx.x
    );
}

extern "C" __global__ void markov_blanket_partition_bfloat16(
    const unsigned short* adjacency,
    const int* internalNodes,
    int* out,
    unsigned int nodeCount,
    unsigned int internalCount
) {
    unsigned int nodeIndex = blockIdx.x * blockDim.x + threadIdx.x;
    markov_blanket_partition_kernel<BFloat16HawkesMarkovStorage, unsigned short>(
        adjacency,
        internalNodes,
        out,
        nodeCount,
        internalCount,
        nodeIndex
    );
}

extern "C" __global__ void markov_flow_bfloat16(
    const unsigned short* mutualInformation,
    const int* partition,
    unsigned short* out,
    unsigned int nodeCount,
    int targetLabel
) {
    unsigned int nodeIndex = blockIdx.x * blockDim.x + threadIdx.x;
    markov_flow_kernel<BFloat16HawkesMarkovStorage, unsigned short>(
        mutualInformation,
        partition,
        out,
        nodeCount,
        targetLabel,
        nodeIndex
    );
}
