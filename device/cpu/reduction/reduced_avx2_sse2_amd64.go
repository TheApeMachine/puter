//go:build amd64

package reduction

//go:noescape
func SumBFloat16SSE2Asm(values *uint16, count int) uint16

//go:noescape
func SumFloat16SSE2Asm(values *uint16, count int) uint16

func SumBF16AVX2(values *uint16, count int) uint16 {
	return SumBF16AVX512(values, count)
}

func SumFP16AVX2(values *uint16, count int) uint16 {
	return SumFP16AVX512(values, count)
}

func SumBF16SSE2(values *uint16, count int) uint16 {
	if count == 0 {
		return 0
	}

	return SumBFloat16SSE2Asm(values, count)
}

func SumFP16SSE2(values *uint16, count int) uint16 {
	if count == 0 {
		return 0
	}

	return SumFloat16SSE2Asm(values, count)
}
