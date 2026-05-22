//go:build amd64

package rope

//go:noescape
func RopePairsFloat32SSE2Asm(out, in, cos, sin *float32, pairs int)

func ropePairsF32SSE2(out, in, cosBuf, sinBuf []float32) {
	pairCount := len(cosBuf)

	if pairCount == 0 {
		return
	}

	RopePairsFloat32SSE2Asm(
		&out[0], &in[0], &cosBuf[0], &sinBuf[0], pairCount,
	)
}
