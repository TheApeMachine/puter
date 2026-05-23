#ifndef PUTER_DEVICE_CUDA_DROPOUT_DROPOUT_CUH
#define PUTER_DEVICE_CUDA_DROPOUT_DROPOUT_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

struct DropoutSeed {
    unsigned int x;
    unsigned int y;
    unsigned int z;
    unsigned int w;
};

static __device__ __forceinline__ unsigned int dropout_xorshift_advance(unsigned int state, unsigned int step) {
    unsigned int value = state;

    for (unsigned int index = 0; index < step; index++) {
        value ^= value << 13u;
        value ^= value >> 17u;
        value ^= value << 5u;
    }

    return value;
}

static __device__ __forceinline__ unsigned int dropout_seed_for_index(
    unsigned int index,
    unsigned int count,
    DropoutSeed seed
) {
    unsigned int blockCount = count & ~3u;

    if (index < blockCount) {
        unsigned int lane = index & 3u;
        unsigned int step = (index >> 2u) + 1u;

        if (lane == 0u) {
            return dropout_xorshift_advance(seed.x, step);
        }

        if (lane == 1u) {
            return dropout_xorshift_advance(seed.y, step);
        }

        if (lane == 2u) {
            return dropout_xorshift_advance(seed.z, step);
        }

        return dropout_xorshift_advance(seed.w, step);
    }

    unsigned int tailStep = (blockCount >> 2u) + (index - blockCount) + 1u;
    return dropout_xorshift_advance(seed.x, tailStep);
}

#endif
