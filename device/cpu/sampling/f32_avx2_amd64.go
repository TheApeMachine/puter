//go:build amd64

package sampling

//go:noescape
func GreedySampleFloat32AVX2Asm(logits *float32, count int) int32

//go:noescape
func SamplingSoftmaxRowFloat32AVX2Asm(logits, out *float32, temperature float32, count int)

func GreedySampleF32AVX2(logits *float32, count int) int32 {
	if count == 0 {
		return 0
	}

	return GreedySampleFloat32AVX2Asm(logits, count)
}

func SamplingSoftmaxRowF32AVX2(logits, out *float32, temperature float32, count int) {
	if count == 0 {
		return
	}

	SamplingSoftmaxRowFloat32AVX2Asm(logits, out, temperature, count)
}
