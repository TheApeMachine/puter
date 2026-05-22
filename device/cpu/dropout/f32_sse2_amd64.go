//go:build amd64

package dropout

//go:noescape
func DropoutFloat32SSE2Asm(
	dst, src *float32,
	count int,
	seedLane *uint32,
	scale, threshold float32,
)

func DropoutF32SSE2(
	dst, src *float32,
	count int,
	seedState *[4]uint32,
	keepProb float32,
) {
	if count == 0 {
		return
	}

	scale := float32(1.0 / keepProb)
	threshold := dropoutThreshold(keepProb)

	DropoutFloat32SSE2Asm(
		dst, src, count,
		&seedState[0], scale, threshold,
	)
}
