//go:build amd64

package reduction

//go:noescape
func MinBFloat16AVX2Asm(values *uint16, count int) float32

//go:noescape
func MaxBFloat16AVX2Asm(values *uint16, count int) float32

//go:noescape
func L1NormBFloat16AVX2Asm(values *uint16, count int) float32

//go:noescape
func MinFloat16AVX2Asm(values *uint16, count int) float32

//go:noescape
func MaxFloat16AVX2Asm(values *uint16, count int) float32

//go:noescape
func L1NormFloat16AVX2Asm(values *uint16, count int) float32

//go:noescape
func MinBFloat16SSE2Asm(values *uint16, count int) float32

//go:noescape
func MaxBFloat16SSE2Asm(values *uint16, count int) float32

//go:noescape
func L1NormBFloat16SSE2Asm(values *uint16, count int) float32

//go:noescape
func MinFloat16SSE2Asm(values *uint16, count int) float32

//go:noescape
func MaxFloat16SSE2Asm(values *uint16, count int) float32

//go:noescape
func L1NormFloat16SSE2Asm(values *uint16, count int) float32

func MinBF16AVX2(values *uint16, count int) float32 {
	if count == 0 {
		return 0
	}

	return MinBFloat16AVX2Asm(values, count)
}

func MaxBF16AVX2(values *uint16, count int) float32 {
	if count == 0 {
		return 0
	}

	return MaxBFloat16AVX2Asm(values, count)
}

func L1NormBF16AVX2(values *uint16, count int) float32 {
	if count == 0 {
		return 0
	}

	return L1NormBFloat16AVX2Asm(values, count)
}

func MinFP16AVX2(values *uint16, count int) float32 {
	if count == 0 {
		return 0
	}

	return MinFloat16AVX2Asm(values, count)
}

func MaxFP16AVX2(values *uint16, count int) float32 {
	if count == 0 {
		return 0
	}

	return MaxFloat16AVX2Asm(values, count)
}

func L1NormFP16AVX2(values *uint16, count int) float32 {
	if count == 0 {
		return 0
	}

	return L1NormFloat16AVX2Asm(values, count)
}

func MinBF16SSE2(values *uint16, count int) float32 {
	if count == 0 {
		return 0
	}

	return MinBFloat16SSE2Asm(values, count)
}

func MaxBF16SSE2(values *uint16, count int) float32 {
	if count == 0 {
		return 0
	}

	return MaxBFloat16SSE2Asm(values, count)
}

func L1NormBF16SSE2(values *uint16, count int) float32 {
	if count == 0 {
		return 0
	}

	return L1NormBFloat16SSE2Asm(values, count)
}

func MinFP16SSE2(values *uint16, count int) float32 {
	if count == 0 {
		return 0
	}

	return MinFloat16SSE2Asm(values, count)
}

func MaxFP16SSE2(values *uint16, count int) float32 {
	if count == 0 {
		return 0
	}

	return MaxFloat16SSE2Asm(values, count)
}

func L1NormFP16SSE2(values *uint16, count int) float32 {
	if count == 0 {
		return 0
	}

	return L1NormFloat16SSE2Asm(values, count)
}
