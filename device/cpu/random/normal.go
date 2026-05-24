package random

/*
NormalFloat32Scalar writes `count` standard-normal float32 values to
`out`, seeded by (seed, counter). Output is bitwise reproducible across
backends given the same inputs (see package doc for the contract).

Implementation: Philox-4×32-10 advances the counter, each Philox call
produces four uniform float32s, every pair of uniforms yields a pair of
Gaussian float32s via Box-Muller. Counter increments by 1 per Philox
call so a single uint64 counter spans 2^64 distinct 128-bit blocks
(8 × 2^64 = 2^67 random float32s before wraparound).

This is the scalar reference. NEON, Metal, CUDA, and XLA implementations
must match this function bitwise for every (seed, counter, count) input.
*/
func NormalFloat32Scalar(out []float32, count int, seed, counter uint64) {
	if count <= 0 {
		return
	}

	if len(out) < count {
		panic("random: out buffer too small for count")
	}

	cursor := 0

	for cursor < count {
		state := NewPhiloxState(seed, counter)
		w0, w1, w2, w3 := Philox4x32(state)
		counter++

		u0 := uniformFloat32(w0)
		u1 := uniformFloat32(w1)
		u2 := uniformFloat32(w2)
		u3 := uniformFloat32(w3)

		z0, z1 := boxMullerPair(u0, u1)
		z2, z3 := boxMullerPair(u2, u3)

		gaussians := [4]float32{z0, z1, z2, z3}

		for index := 0; index < 4 && cursor < count; index++ {
			out[cursor] = gaussians[index]
			cursor++
		}
	}
}
