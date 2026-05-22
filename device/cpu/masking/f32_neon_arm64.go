//go:build arm64

package masking

import "unsafe"

//go:noescape
func ApplyMaskFloat32NEONAsm(input, mask, output *float32, count int)

//go:noescape
func causalMaskFloat32NEONFillAsm(rowOutput *float32, zeroCount, infCount int)

//go:noescape
func alibiBiasFloat32NEONElemAsm(score, slope, output *float32, distance int)

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

	for rowIndex := 0; rowIndex < seqQ; rowIndex++ {
		zeroCount := rowIndex + 1
		if zeroCount > seqK {
			zeroCount = seqK
		}

		infCount := seqK - zeroCount
		rowOutput := (*float32)(unsafe.Add(
			unsafe.Pointer(output),
			rowIndex*seqK*4,
		))
		causalMaskFloat32NEONFillAsm(rowOutput, zeroCount, infCount)
	}
}

func ALiBiBiasF32NEON(scores, slope, output *float32, seqQ, seqK int) {
	if seqQ == 0 || seqK == 0 {
		return
	}

	for rowIndex := 0; rowIndex < seqQ; rowIndex++ {
		for colIndex := 0; colIndex < seqK; colIndex++ {
			index := rowIndex*seqK + colIndex
			distance := rowIndex - colIndex
			score := (*float32)(unsafe.Add(unsafe.Pointer(scores), index*4))
			out := (*float32)(unsafe.Add(unsafe.Pointer(output), index*4))
			alibiBiasFloat32NEONElemAsm(score, slope, out, distance)
		}
	}
}
