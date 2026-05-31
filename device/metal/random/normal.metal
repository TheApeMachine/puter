// Standalone Metal kernel for random_normal_float32. Each metallibgen
// translation unit is compiled independently, so we keep all constants,
// helpers, and the kernel itself in one file rather than relying on
// cross-file includes (the {family}.metal hub pattern from §2.3.1 is
// declared as convention but not actually used in practice — see
// reduction/aggregate.metal for the precedent).

#include <metal_stdlib>
#include "../elementwise/elementwise_f64_transcendental.metalinc"

using namespace metal;

// Philox-4×32-10 constants. Must match device/cpu/random/philox.go
// bitwise; do not change.
constant uint PHILOX_M0 = 0xD2511F53u;
constant uint PHILOX_M1 = 0xCD9E8D57u;
constant uint PHILOX_W0 = 0x9E3779B9u;
constant uint PHILOX_W1 = 0xBB67AE85u;

constant float SMALLEST_POSITIVE_F32 = 0x1.0p-23f;

/*
philox_round applies one Philox-4×32 round. Matches Philox4x32 in
device/cpu/random/philox.go bitwise.
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
philox4x32_10 produces four pseudorandom uint32s. Bitwise equivalent to
Philox4x32 in device/cpu/random/philox.go.
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
[0, 1) using the mantissa-stuffing trick. Bitwise equivalent to
uniformFloat32 in device/cpu/random/boxmuller.go.
*/
inline float uniform_from_bits(uint bits) {
    uint mantissa = bits >> 9;
    return as_type<float>(0x3F800000u | mantissa) - 1.0f;
}

/*
random_normal_float32 is the Metal kernel that produces standard-normal
float32 samples. Each thread handles one Philox-4×32-10 block, producing
four Gaussian outputs from a single (seed, counter+threadID) pair.

The kernel's Philox output is bitwise equivalent to the CPU scalar
reference. Box-Muller uses SF64 softfloat log/sqrt/sincos matching Go's
float64-then-cast scalar reference in device/cpu/random/boxmuller.go.

Buffer layout:
  [[buffer(0)]] device float *out  - destination, length ≥ count
  [[buffer(1)]] count              - number of Gaussian outputs
  [[buffer(2)]] seedLo             - low 32 bits of seed
  [[buffer(3)]] seedHi             - high 32 bits of seed
  [[buffer(4)]] ctrLo              - low 32 bits of base counter
  [[buffer(5)]] ctrHi              - high 32 bits of base counter

Counter mapping: thread `t` uses counter `(ctrBase + t)`. Caller must
ensure `ctrLo + threadCount` fits in 32 bits (no carry propagation into
ctrHi); for very large output buffers split the work across launches
with explicit ctrHi increments.
*/
inline float2 box_muller_pair(float uniformFirst, float uniformSecond) {
    return metal_sf64_box_muller_pair(uniformFirst, uniformSecond);
}

kernel void random_normal_float32(
    device float *out             [[buffer(0)]],
    constant uint &count          [[buffer(1)]],
    constant uint &seedLo         [[buffer(2)]],
    constant uint &seedHi         [[buffer(3)]],
    constant uint &ctrLo          [[buffer(4)]],
    constant uint &ctrHi          [[buffer(5)]],
    uint thread_id [[thread_position_in_grid]]
) {
    uint baseIdx = thread_id * 4u;

    if (baseIdx >= count) {
        return;
    }

    uint w0, w1, w2, w3;
    philox4x32_10(seedLo, seedHi, ctrLo + thread_id, ctrHi, w0, w1, w2, w3);

    float u0 = uniform_from_bits(w0);
    float u1 = uniform_from_bits(w1);
    float u2 = uniform_from_bits(w2);
    float u3 = uniform_from_bits(w3);

    if (u0 == 0.0f) {
        u0 = SMALLEST_POSITIVE_F32;
    }
    if (u2 == 0.0f) {
        u2 = SMALLEST_POSITIVE_F32;
    }

    float2 pair0 = box_muller_pair(u0, u1);
    float2 pair1 = box_muller_pair(u2, u3);

    if (baseIdx + 0u < count) out[baseIdx + 0u] = pair0.x;
    if (baseIdx + 1u < count) out[baseIdx + 1u] = pair0.y;
    if (baseIdx + 2u < count) out[baseIdx + 2u] = pair1.x;
    if (baseIdx + 3u < count) out[baseIdx + 3u] = pair1.y;
}
