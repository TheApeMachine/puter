//go:build arm64

package sampling

//go:noescape
func GreedySampleFloat32NEONAsm(logits *float32, count int) int32

//go:noescape
func SamplingSoftmaxRowFloat32NEONAsm(logits, out *float32, temperature float32, count int)

func GreedySampleF32NEON(logits *float32, count int) int32 {
	if count == 0 {
		return 0
	}

	return GreedySampleFloat32NEONAsm(logits, count)
}

func SamplingSoftmaxRowF32NEON(logits, out *float32, temperature float32, count int) {
	if count == 0 {
		return
	}

	SamplingSoftmaxRowFloat32NEONAsm(logits, out, temperature, count)
}
