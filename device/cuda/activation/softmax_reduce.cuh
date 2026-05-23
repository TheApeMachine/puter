#ifndef PUTER_DEVICE_CUDA_ACTIVATION_SOFTMAX_REDUCE_CUH
#define PUTER_DEVICE_CUDA_ACTIVATION_SOFTMAX_REDUCE_CUH

static __device__ __forceinline__ float softmax_warp_reduce_max(float value) {
    for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
        float other = __shfl_down_sync(0xffffffffu, value, offset);
        value = fmaxf(value, other);
    }

    return value;
}

static __device__ __forceinline__ float softmax_warp_reduce_sum(float value) {
    for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }

    return value;
}

static __device__ __forceinline__ float softmax_block_reduce_max(float value, float* shared) {
    unsigned int lane = threadIdx.x & 31u;
    unsigned int warp = threadIdx.x >> 5;
    unsigned int warpCount = (blockDim.x + 31u) >> 5;

    value = softmax_warp_reduce_max(value);

    if (lane == 0u) {
        shared[warp] = value;
    }

    __syncthreads();

    if (warp == 0u) {
        value = lane < warpCount ? shared[lane] : -CUDART_INF_F;
        value = softmax_warp_reduce_max(value);
    }

    return __shfl_sync(0xffffffffu, value, 0);
}

static __device__ __forceinline__ float softmax_block_reduce_sum(float value, float* shared) {
    unsigned int lane = threadIdx.x & 31u;
    unsigned int warp = threadIdx.x >> 5;
    unsigned int warpCount = (blockDim.x + 31u) >> 5;

    value = softmax_warp_reduce_sum(value);

    if (lane == 0u) {
        shared[warp] = value;
    }

    __syncthreads();

    if (warp == 0u) {
        value = lane < warpCount ? shared[lane] : 0.0f;
        value = softmax_warp_reduce_sum(value);
    }

    return __shfl_sync(0xffffffffu, value, 0);
}

#endif
