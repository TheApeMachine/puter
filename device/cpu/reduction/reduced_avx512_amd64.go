//go:build amd64

package reduction

//go:noescape
func SumBFloat16AVX512Asm(values *uint16, count int) uint16

//go:noescape
func SumFloat16AVX512Asm(values *uint16, count int) uint16

func SumBF16AVX512(values *uint16, count int) uint16 {
	if count == 0 {
		return 0
	}

	return SumBFloat16AVX512Asm(values, count)
}

func SumFP16AVX512(values *uint16, count int) uint16 {
	if count == 0 {
		return 0
	}

	return SumFloat16AVX512Asm(values, count)
}
