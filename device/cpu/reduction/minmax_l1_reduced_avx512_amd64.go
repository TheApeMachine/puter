//go:build amd64

package reduction

//go:noescape
func MinBFloat16AVX512Asm(values *uint16, count int) float32

//go:noescape
func MaxBFloat16AVX512Asm(values *uint16, count int) float32

//go:noescape
func L1NormBFloat16AVX512Asm(values *uint16, count int) float32

//go:noescape
func MinFloat16AVX512Asm(values *uint16, count int) float32

//go:noescape
func MaxFloat16AVX512Asm(values *uint16, count int) float32

//go:noescape
func L1NormFloat16AVX512Asm(values *uint16, count int) float32

func MinBF16AVX512(values *uint16, count int) float32 {
	if count == 0 {
		return 0
	}

	return MinBFloat16AVX512Asm(values, count)
}

func MaxBF16AVX512(values *uint16, count int) float32 {
	if count == 0 {
		return 0
	}

	return MaxBFloat16AVX512Asm(values, count)
}

func L1NormBF16AVX512(values *uint16, count int) float32 {
	if count == 0 {
		return 0
	}

	return L1NormBFloat16AVX512Asm(values, count)
}

func MinFP16AVX512(values *uint16, count int) float32 {
	if count == 0 {
		return 0
	}

	return MinFloat16AVX512Asm(values, count)
}

func MaxFP16AVX512(values *uint16, count int) float32 {
	if count == 0 {
		return 0
	}

	return MaxFloat16AVX512Asm(values, count)
}

func L1NormFP16AVX512(values *uint16, count int) float32 {
	if count == 0 {
		return 0
	}

	return L1NormFloat16AVX512Asm(values, count)
}
