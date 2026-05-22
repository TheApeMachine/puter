//go:build amd64

package sampling

//go:noescape
func GreedySampleFloat32SSE2Asm(logits *float32, count int) int32

//go:noescape
func SamplingSoftmaxRowFloat32SSE2Asm(logits, out *float32, temperature float32, count int)

func GreedySampleF32SSE2(logits *float32, count int) int32 {
	if count == 0 {
		return 0
	}

	return GreedySampleFloat32SSE2Asm(logits, count)
}

func SamplingSoftmaxRowF32SSE2(logits, out *float32, temperature float32, count int) {
	if count == 0 {
		return
	}

	SamplingSoftmaxRowFloat32SSE2Asm(logits, out, temperature, count)
}
