#include <metal_stdlib>
#include "dropout_jump.h"

using namespace metal;

static inline uint dropout_seed_for_index(uint index, uint count, uint4 seed) {
    uint blockCount = count & ~3u;

    if (index < blockCount) {
        uint lane = index & 3u;
        uint step = (index >> 2u) + 1u;

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

    uint tailStep = (blockCount >> 2u) + (index - blockCount) + 1u;
    return dropout_xorshift_advance(seed.x, tailStep);
}

#define DROPOUT_KERNEL(name, scalar, stored_type, load_expr, store_expr) \
kernel void name( \
    device const scalar* input [[buffer(0)]], \
    device scalar* out [[buffer(1)]], \
    constant uint& count [[buffer(2)]], \
    constant float& scale [[buffer(3)]], \
    constant uint& threshold [[buffer(4)]], \
    constant uint4& seed [[buffer(5)]], \
    uint index [[thread_position_in_grid]] \
) { \
    if (index >= count) { \
        return; \
    } \
    uint randomValue = dropout_seed_for_index(index, count, seed); \
    stored_type outValue = stored_type(0); \
    if (randomValue < threshold) { \
        outValue = load_expr * stored_type(scale); \
    } \
    store_expr; \
}

DROPOUT_KERNEL(
    dropout_float32,
    float,
    float,
    input[index],
    out[index] = outValue
)

DROPOUT_KERNEL(
    dropout_float16,
    half,
    half,
    input[index],
    out[index] = outValue
)

DROPOUT_KERNEL(
    dropout_bfloat16,
    ushort,
    bfloat,
    as_type<bfloat>(input[index]),
    out[index] = as_type<ushort>(outValue)
)
