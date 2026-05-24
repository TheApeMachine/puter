// Standalone Metal kernel for random_normal_float32. Each metallibgen
// translation unit is compiled independently, so we keep all constants,
// helpers, and the kernel itself in one file rather than relying on
// cross-file includes (the {family}.metal hub pattern from §2.3.1 is
// declared as convention but not actually used in practice — see
// reduction/aggregate.metal for the precedent).

#include <metal_stdlib>

using namespace metal;

// Philox-4×32-10 constants. Must match device/cpu/random/philox.go
// bitwise; do not change.
constant uint PHILOX_M0 = 0xD2511F53u;
constant uint PHILOX_M1 = 0xCD9E8D57u;
constant uint PHILOX_W0 = 0x9E3779B9u;
constant uint PHILOX_W1 = 0xBB67AE85u;

constant float TWO_PI_F = 6.28318530717958647692f;
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
reference. Box-Muller uses Metal's native single-precision log, sin,
cos, sqrt — these do not bitwise match Go's F64-then-cast scalar
reference, so parity tests assert ≤ 8 ULP per lane on the final
Gaussians, not bitwise equality.

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

    // Box-Muller degenerate-case guard matches the scalar reference:
    // when u_first == 0, substitute 2^-23 so log() stays finite.
    if (u0 == 0.0f) {
        u0 = SMALLEST_POSITIVE_F32;
    }
    if (u2 == 0.0f) {
        u2 = SMALLEST_POSITIVE_F32;
    }

    // Use metal::precise::* transcendentals rather than the default
    // (fast) variants. The defaults can drift several ULPs from the
    // correctly-rounded F32 result, which compounds through the
    // magnitude × sin/cos multiply and shows up as 50+ ULP gaps on
    // small-result lanes. The precise::* variants are spec'd at
    // ≤ 1 ULP from correctly-rounded F32, which keeps the final
    // Gaussian within the 8 ULP parity budget.
    float magnitude0 = precise::sqrt(-2.0f * precise::log(u0));
    float angle0 = TWO_PI_F * u1;
    float magnitude1 = precise::sqrt(-2.0f * precise::log(u2));
    float angle1 = TWO_PI_F * u3;

    float sin0 = precise::sin(angle0);
    float cos0 = precise::cos(angle0);
    float sin1 = precise::sin(angle1);
    float cos1 = precise::cos(angle1);

    float z0 = magnitude0 * cos0;
    float z1 = magnitude0 * sin0;
    float z2 = magnitude1 * cos1;
    float z3 = magnitude1 * sin1;

    if (baseIdx + 0u < count) out[baseIdx + 0u] = z0;
    if (baseIdx + 1u < count) out[baseIdx + 1u] = z1;
    if (baseIdx + 2u < count) out[baseIdx + 2u] = z2;
    if (baseIdx + 3u < count) out[baseIdx + 3u] = z3;
}
