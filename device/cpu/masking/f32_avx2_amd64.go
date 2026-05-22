//go:build amd64

package masking

//go:noescape
func ApplyMaskFloat32AVX2Asm(input, mask, output *float32, count int)

//go:noescape
func CausalMaskFloat32AVX2Asm(output *float32, seqQ, seqK int)

//go:noescape
func ALiBiBiasFloat32AVX2Asm(scores, slope, output *float32, seqQ, seqK int)

func ApplyMaskF32AVX2(input, mask, output *float32, count int) {
	if count == 0 {
		return
	}

	ApplyMaskFloat32AVX2Asm(input, mask, output, count)
}

func CausalMaskF32AVX2(output *float32, seqQ, seqK int) {
	if seqQ == 0 || seqK == 0 {
		return
	}

	CausalMaskFloat32AVX2Asm(output, seqQ, seqK)
}

func applyMaskF32AVX2(input, mask, output *float32, count int) {
	ApplyMaskF32AVX2(input, mask, output, count)
}

func causalMaskF32AVX2(output *float32, seqQ, seqK int) {
	CausalMaskF32AVX2(output, seqQ, seqK)
}

func ALiBiBiasF32AVX2(scores, slope, output *float32, seqQ, seqK int) {
	if seqQ == 0 || seqK == 0 {
		return
	}

	ALiBiBiasFloat32AVX2Asm(scores, slope, output, seqQ, seqK)
}

func alibiBiasF32AVX2(scores, slope, output *float32, seqQ, seqK int) {
	ALiBiBiasF32AVX2(scores, slope, output, seqQ, seqK)
}
