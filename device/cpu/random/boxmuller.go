package random

import "math"

/*
uniformFloat32 converts a 32-bit random word into a uniform float32 in
the half-open interval [0, 1). The conversion uses the standard
"mantissa-stuffing" trick: take the top 23 random bits of the input,
place them in the mantissa of 1.0 (exponent bias 127), then subtract
1.0.

This is the canonical SIMD-friendly uniform conversion because it is
branchless, requires no division, and yields exactly 2^23 distinct
float32 values uniformly distributed across [0, 1) — every
representable float32 in the interval at single-precision granularity.
*/
func uniformFloat32(bits uint32) float32 {
	const oneAsBits = uint32(0x3F800000) // float32(1.0)
	mantissa := bits >> 9
	return math.Float32frombits(oneAsBits|mantissa) - 1.0
}

/*
boxMullerPair converts a pair of uniform [0, 1) float32 values into a
pair of standard-normal float32 values via the classical Box-Muller
transform:

	magnitude = sqrt(-2 ln u1)
	angle     = 2π u2
	z0        = magnitude · cos(angle)
	z1        = magnitude · sin(angle)

Intermediate computations promote to float64 for log/sqrt/sincos
because Go's stdlib only exposes those at float64 precision. The final
results are cast back to float32. Metal/CUDA/XLA kernels must perform
the same precision sequence (precise float64 ln/sqrt/sincos, then
round to float32) to maintain bitwise parity.

Edge case: when u1 == 0, ln(u1) is -∞ and magnitude is +∞. The function
substitutes the smallest positive float32 (2^-23) for u1 in that case.
This is deterministic, hits with probability ≈ 2^-23 (only when the top
23 random bits of the input word are all zero), and keeps the output
finite.
*/
func boxMullerPair(u1, u2 float32) (float32, float32) {
	if u1 == 0 {
		u1 = math.Float32frombits(uint32(0x34000000)) // 2^-23
	}

	magnitude := math.Sqrt(-2.0 * math.Log(float64(u1)))
	sin, cos := math.Sincos(2.0 * math.Pi * float64(u2))

	return float32(magnitude * cos), float32(magnitude * sin)
}
