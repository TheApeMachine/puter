// Package random implements counter-based pseudorandom number
// generation for the CPU backend.
//
// Algorithm: Philox-4×32-10 (Salmon, Moraes, Dror, Shaw, 2011 — see
// "Parallel Random Numbers: As Easy as 1, 2, 3" / the random123 library)
// seeded with a uint64 key and indexed by a uint64 counter. Each Philox
// invocation emits four uint32 words derived deterministically from
// (seed, counter), so two key properties hold:
//
//   - Bitwise reproducibility across backends: a Metal, CUDA, or XLA
//     kernel that performs the same Philox-4×32-10 rounds on the same
//     (seed, counter) input must emit exactly the same four uint32s as
//     the scalar reference in this package. The parity tests in
//     random_parity_test.go pin this contract.
//
//   - Embarrassingly parallel SIMD/GPU implementation: each counter
//     position is independent of every other, so a NEON or Metal kernel
//     can produce 4 (or 16) outputs simultaneously by handing each lane
//     its own counter.
//
// The uniform output of Philox is converted to standard-normal float32
// via the Box-Muller transform (see boxmuller.go). Each Philox call
// yields four uint32s → two uniform pairs → two Gaussian pairs (four
// normal float32 outputs).
//
// This package is intentionally NOT wired to the device.Backend
// interface yet. It is the scalar reference that NEON, Metal, CUDA, and
// XLA implementations must match bitwise; interface declaration lands
// only after every required backend has its own real implementation
// with parity verified (see puter/GAPS.md §6.5).
package random

/*
Random implements counter-based Gaussian random number generation for
the CPU backend.
*/
type Random struct{}

/*
New constructs a Random receiver for CPU dispatch.
*/
func New() Random {
	return Random{}
}

/*
Default is the package-level Random receiver. Callers that do not need
to hold state can use Default directly; tests and parity harnesses
typically construct their own via New().
*/
var Default = New()
