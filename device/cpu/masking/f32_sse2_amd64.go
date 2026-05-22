//go:build amd64

package masking

//go:noescape
func ApplyMaskFloat32SSE2Asm(input, mask, output *float32, count int)

//go:noescape
func CausalMaskFloat32SSE2Asm(output *float32, seqQ, seqK int)

//go:noescape
func ALiBiBiasFloat32SSE2Asm(scores, slope, output *float32, seqQ, seqK int)

func ApplyMaskF32SSE2(input, mask, output *float32, count int) {
	if count == 0 {
		return
	}

	ApplyMaskFloat32SSE2Asm(input, mask, output, count)
}

func CausalMaskF32SSE2(output *float32, seqQ, seqK int) {
	if seqQ == 0 || seqK == 0 {
		return
	}

	CausalMaskFloat32SSE2Asm(output, seqQ, seqK)
}

func applyMaskF32SSE2(input, mask, output *float32, count int) {
	ApplyMaskF32SSE2(input, mask, output, count)
}

func causalMaskF32SSE2(output *float32, seqQ, seqK int) {
	CausalMaskF32SSE2(output, seqQ, seqK)
}

func ALiBiBiasF32SSE2(scores, slope, output *float32, seqQ, seqK int) {
	if seqQ == 0 || seqK == 0 {
		return
	}

	ALiBiBiasFloat32SSE2Asm(scores, slope, output, seqQ, seqK)
}

func alibiBiasF32SSE2(scores, slope, output *float32, seqQ, seqK int) {
	ALiBiBiasF32SSE2(scores, slope, output, seqQ, seqK)
}
