//go:build arm64

package random

/*
Philox4x32x4NEON computes four parallel lanes of Philox-4×32-10 in NEON,
yielding 16 uint32 outputs. Each lane uses counter = ctrBase + laneIndex
(0 ≤ laneIndex < 4); the high 32 bits of each lane's Ctr1 are taken
from `ctrBase >> 32` (no overflow is propagated from Ctr0+lane into
Ctr1, so the caller must ensure ctrBase + 3 fits in the low 32 bits of
the counter — i.e. ctrBaseLow ≤ 0xFFFFFFFC). The kernel matches the
scalar Philox4x32 reference bitwise for every input.

Output layout: [lane0_w0, lane0_w1, lane0_w2, lane0_w3, lane1_w0, ...,
lane3_w3]. This is the interleaved order produced by NEON ST4, and
matches what the scalar reference would emit when called four times in
sequence with counters ctrBase, ctrBase+1, ctrBase+2, ctrBase+3 and
concatenated.

`out` must point to at least 16 uint32 (64 bytes) of writable memory.
*/
//go:noescape
func Philox4x32x4NEON(out *uint32, key0 uint32, key1 uint32, ctrBase uint64)
