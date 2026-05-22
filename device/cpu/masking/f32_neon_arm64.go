//go:build arm64

package masking

//go:noescape
func ApplyMaskFloat32NEONAsm(input, mask, output *float32, count int)

//go:noescape
func CausalMaskFloat32NEONAsm(output *float32, seqQ, seqK int)

//go:noescape
func ALiBiBiasFloat32NEONAsm(scores, slope, output *float32, seqQ, seqK int)

func ApplyMaskF32NEON(input, mask, output *float32, count int) {
	if count == 0 {
		return
	}

	ApplyMaskFloat32NEONAsm(input, mask, output, count)
}

func CausalMaskF32NEON(output *float32, seqQ, seqK int) {
	if seqQ == 0 || seqK == 0 {
		return
	}

	CausalMaskFloat32NEONAsm(output, seqQ, seqK)
}

func ALiBiBiasF32NEON(scores, slope, output *float32, seqQ, seqK int) {
	if seqQ == 0 || seqK == 0 {
		return
	}

	ALiBiBiasFloat32NEONAsm(scores, slope, output, seqQ, seqK)
}
