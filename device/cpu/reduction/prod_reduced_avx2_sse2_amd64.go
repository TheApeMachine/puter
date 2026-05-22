//go:build amd64

package reduction

//go:noescape
func ProdBFloat16AVX2Asm(values *uint16, count int) float32

//go:noescape
func ProdFloat16AVX2Asm(values *uint16, count int) float32

//go:noescape
func ProdBFloat16SSE2Asm(values *uint16, count int) float32

//go:noescape
func ProdFloat16SSE2Asm(values *uint16, count int) float32

func ProdBF16AVX2(values *uint16, count int) float32 {
	if count == 0 {
		return 0
	}

	return ProdBFloat16AVX2Asm(values, count)
}

func ProdFP16AVX2(values *uint16, count int) float32 {
	if count == 0 {
		return 0
	}

	return ProdFloat16AVX2Asm(values, count)
}

func ProdBF16SSE2(values *uint16, count int) float32 {
	if count == 0 {
		return 0
	}

	return ProdBFloat16SSE2Asm(values, count)
}

func ProdFP16SSE2(values *uint16, count int) float32 {
	if count == 0 {
		return 0
	}

	return ProdFloat16SSE2Asm(values, count)
}
