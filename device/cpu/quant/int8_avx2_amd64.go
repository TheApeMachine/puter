//go:build amd64

package quant

//go:noescape
func QuantInt8AVX2Asm(dst *int8, src *float32, count int, invScale float32, zeroPoint int32)

func quantInt8AVX2(dst []int8, src []float32, scale float32, zeroPoint int8) {
	elementCount := len(dst)

	if elementCount == 0 {
		return
	}

	invScale := float32(1.0 / scale)

	QuantInt8AVX2Asm(
		&dst[0], &src[0], elementCount,
		invScale, int32(zeroPoint),
	)
}
