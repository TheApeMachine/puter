//go:build amd64

package reduction

//go:noescape
func ProdBFloat16AVX512Asm(values *uint16, count int) float32

//go:noescape
func ProdFloat16AVX512Asm(values *uint16, count int) float32

func ProdBF16AVX512(values *uint16, count int) float32 {
	if count == 0 {
		return 0
	}

	return ProdBFloat16AVX512Asm(values, count)
}

func ProdFP16AVX512(values *uint16, count int) float32 {
	if count == 0 {
		return 0
	}

	return ProdFloat16AVX512Asm(values, count)
}
