package random

// Philox-4×32-10 constants from Salmon et al. (2011). These are fixed
// and verified against the canonical random123 reference implementation
// (kat_vectors.dat). Do not change them — every backend's parity test
// depends on these exact values producing the exact reference output.
const (
	philoxM4x32_0 = uint32(0xD2511F53)
	philoxM4x32_1 = uint32(0xCD9E8D57)
	philoxW32_0   = uint32(0x9E3779B9) // Weyl: golden ratio
	philoxW32_1   = uint32(0xBB67AE85) // Weyl: sqrt(3) - 1
	philoxRounds  = 10
)

/*
PhiloxState holds the Philox-4×32 key (2× uint32) and counter (4×
uint32). The key derives from a uint64 seed; the counter spans up to
2^128 unique 128-bit blocks but this package's NormalFloat32Scalar
exposes only a uint64 counter for ergonomic reasons, leaving the high
64 bits at zero.

PhiloxState is passed by value through the round function so the caller
can hold a single state and produce many outputs by incrementing the
counter between calls without aliasing.
*/
type PhiloxState struct {
	Key0, Key1             uint32
	Ctr0, Ctr1, Ctr2, Ctr3 uint32
}

/*
NewPhiloxState constructs a Philox state from a uint64 seed and a uint64
counter. The seed splits into (Key0=lo32, Key1=hi32). The counter
splits into (Ctr0=lo32, Ctr1=hi32) with Ctr2 and Ctr3 left at zero, so
a single uint64 counter spans 2^64 distinct 128-bit Philox blocks
(8 × 2^64 random float32s before counter wraparound — effectively
infinite for any practical workload).
*/
func NewPhiloxState(seed, counter uint64) PhiloxState {
	return PhiloxState{
		Key0: uint32(seed),
		Key1: uint32(seed >> 32),
		Ctr0: uint32(counter),
		Ctr1: uint32(counter >> 32),
		Ctr2: 0,
		Ctr3: 0,
	}
}

/*
mulhilo32 returns the high and low 32 bits of the 64-bit unsigned
product a × b. This is the only non-XOR/non-shift primitive Philox
needs; SIMD implementations must produce identical results by promoting
to 64-bit before multiplying.
*/
func mulhilo32(a, b uint32) (high, low uint32) {
	product := uint64(a) * uint64(b)
	return uint32(product >> 32), uint32(product)
}

/*
philoxRound applies one round of the Philox-4×32 permutation to the
counter words (c0, c1, c2, c3) using key words (k0, k1). The round
function multiplies two of the counter words by the M constants,
splits each 64-bit product into high and low halves, then crosses the
products with the surviving counter words and the key.
*/
func philoxRound(c0, c1, c2, c3, k0, k1 uint32) (uint32, uint32, uint32, uint32) {
	hi0, lo0 := mulhilo32(philoxM4x32_0, c0)
	hi1, lo1 := mulhilo32(philoxM4x32_1, c2)

	return hi1 ^ c1 ^ k0,
		lo1,
		hi0 ^ c3 ^ k1,
		lo0
}

/*
Philox4x32 produces four pseudorandom uint32s from a Philox state. The
state is consumed by value, so the caller's local copy is unchanged;
to produce more output, the caller increments the counter and calls
again.

This is the canonical reference. NEON, Metal, CUDA, and XLA
implementations must match this function bitwise for every (seed,
counter) input. The parity tests in random_parity_test.go pin three
representative input/output pairs from the random123 kat_vectors file.
*/
func Philox4x32(state PhiloxState) (uint32, uint32, uint32, uint32) {
	c0, c1, c2, c3 := state.Ctr0, state.Ctr1, state.Ctr2, state.Ctr3
	k0, k1 := state.Key0, state.Key1

	for range philoxRounds {
		c0, c1, c2, c3 = philoxRound(c0, c1, c2, c3, k0, k1)
		k0 += philoxW32_0
		k1 += philoxW32_1
	}

	return c0, c1, c2, c3
}
