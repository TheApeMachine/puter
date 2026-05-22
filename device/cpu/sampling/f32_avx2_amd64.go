//go:build amd64

package sampling

//go:noescape
func GreedySampleFloat32AVX2Asm(logits *float32, count int) int32

func GreedySampleF32AVX2(logits *float32, count int) int32 {
	if count == 0 {
		return 0
	}

	return GreedySampleFloat32AVX2Asm(logits, count)
}
