#include "random.metal"

using namespace metal;

constant float TWO_PI_F = 6.28318530717958647692f;
constant float SMALLEST_POSITIVE_F32 = 0x1.0p-23f;

/*
random_normal_float32 is the Metal kernel that produces standard-normal
float32 samples. Each thread handles one Philox-4×32-10 block, producing
four Gaussian outputs from a single (seed, counter+threadID) pair.

The kernel's Philox output is bitwise equivalent to the CPU scalar
reference. Box-Muller uses Metal's native single-precision log, sin,
cos, sqrt — these do not bitwise match Go's F64-then-cast scalar
reference, so parity tests assert ≤ 4 ULP per lane on the final
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

    float magnitude0 = sqrt(-2.0f * log(u0));
    float angle0 = TWO_PI_F * u1;
    float magnitude1 = sqrt(-2.0f * log(u2));
    float angle1 = TWO_PI_F * u3;

    float sin0 = sin(angle0);
    float cos0 = cos(angle0);
    float sin1 = sin(angle1);
    float cos1 = cos(angle1);

    float z0 = magnitude0 * cos0;
    float z1 = magnitude0 * sin0;
    float z2 = magnitude1 * cos1;
    float z3 = magnitude1 * sin1;

    if (baseIdx + 0u < count) out[baseIdx + 0u] = z0;
    if (baseIdx + 1u < count) out[baseIdx + 1u] = z1;
    if (baseIdx + 2u < count) out[baseIdx + 2u] = z2;
    if (baseIdx + 3u < count) out[baseIdx + 3u] = z3;
}
