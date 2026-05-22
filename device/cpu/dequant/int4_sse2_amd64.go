//go:build amd64

package dequant

//go:noescape
func DequantInt4SSE2Asm(dst *float32, src *byte, count int, scale float32, zeroPoint int8)

func dequantInt4SSE2(dst []float32, src []byte, elementCount int, scale float32, zeroPoint int8) {
	if elementCount == 0 {
		return
	}

	DequantInt4SSE2Asm(&dst[0], &src[0], elementCount, scale, zeroPoint)
}
