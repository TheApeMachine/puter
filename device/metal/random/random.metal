#ifndef PUTER_DEVICE_METAL_RANDOM_RANDOM_METAL
#define PUTER_DEVICE_METAL_RANDOM_RANDOM_METAL

#include <metal_stdlib>

using namespace metal;

// Philox-4×32-10 constants. Must match device/cpu/random/philox.go
// bitwise; do not change.
constant uint PHILOX_M0 = 0xD2511F53u;
constant uint PHILOX_M1 = 0xCD9E8D57u;
constant uint PHILOX_W0 = 0x9E3779B9u;
constant uint PHILOX_W1 = 0xBB67AE85u;

/*
philox_round applies one Philox-4×32 round to (c0, c1, c2, c3) keyed by
(k0, k1). Matches the scalar reference in device/cpu/random/philox.go.
*/
inline void philox_round(
    thread uint &c0,
    thread uint &c1,
    thread uint &c2,
    thread uint &c3,
    uint k0,
    uint k1
) {
    ulong product0 = ulong(PHILOX_M0) * ulong(c0);
    ulong product1 = ulong(PHILOX_M1) * ulong(c2);

    uint hi0 = uint(product0 >> 32);
    uint lo0 = uint(product0);
    uint hi1 = uint(product1 >> 32);
    uint lo1 = uint(product1);

    uint new_c0 = hi1 ^ c1 ^ k0;
    uint new_c2 = hi0 ^ c3 ^ k1;
    c0 = new_c0;
    c1 = lo1;
    c2 = new_c2;
    c3 = lo0;
}

/*
philox4x32_10 produces four pseudorandom uint32s from (seed, counter).
Bitwise equivalent to Philox4x32 in device/cpu/random/philox.go.
*/
inline void philox4x32_10(
    uint seedLo,
    uint seedHi,
    uint ctrLo,
    uint ctrHi,
    thread uint &w0,
    thread uint &w1,
    thread uint &w2,
    thread uint &w3
) {
    uint c0 = ctrLo;
    uint c1 = ctrHi;
    uint c2 = 0u;
    uint c3 = 0u;
    uint k0 = seedLo;
    uint k1 = seedHi;

    for (int round = 0; round < 10; round++) {
        philox_round(c0, c1, c2, c3, k0, k1);
        k0 += PHILOX_W0;
        k1 += PHILOX_W1;
    }

    w0 = c0;
    w1 = c1;
    w2 = c2;
    w3 = c3;
}

/*
uniform_from_bits converts a 32-bit random word into a uniform float in
[0, 1) using the mantissa-stuffing trick (top 23 bits become mantissa
of 1.0, subtract 1.0). Bitwise equivalent to uniformFloat32 in
device/cpu/random/boxmuller.go.
*/
inline float uniform_from_bits(uint bits) {
    uint mantissa = bits >> 9;
    return as_type<float>(0x3F800000u | mantissa) - 1.0f;
}

#endif
