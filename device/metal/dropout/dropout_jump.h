#ifndef PUTER_DEVICE_METAL_DROPOUT_DROPOUT_JUMP_H
#define PUTER_DEVICE_METAL_DROPOUT_DROPOUT_JUMP_H

#include <metal_stdlib>
using namespace metal;

static inline uint dropout_xorshift_advance(uint state, uint step) {
    uint value = state;

    for (uint index = 0; index < step; index++) {
        value ^= value << 13u;
        value ^= value >> 17u;
        value ^= value << 5u;
    }

    return value;
}

#endif
