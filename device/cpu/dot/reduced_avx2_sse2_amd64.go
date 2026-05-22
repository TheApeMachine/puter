//go:build amd64

package dot

//go:noescape
func DotBFloat16SSE2Asm(left, right *uint16, count int) uint16

//go:noescape
func DotFloat16SSE2Asm(left, right *uint16, count int) uint16

func DotBF16AVX2(left, right *uint16, count int) uint16 {
	return DotBF16AVX512(left, right, count)
}

func DotFP16AVX2(left, right *uint16, count int) uint16 {
	return DotFP16AVX512(left, right, count)
}

func DotInt8AVX2(left, right *int8, count int) int32 {
	return DotInt8AVX512(left, right, count)
}

func DotBF16SSE2(left, right *uint16, count int) uint16 {
	if count == 0 {
		return 0
	}

	return DotBFloat16SSE2Asm(left, right, count)
}

func DotFP16SSE2(left, right *uint16, count int) uint16 {
	if count == 0 {
		return 0
	}

	return DotFloat16SSE2Asm(left, right, count)
}

func DotInt8SSE2(left, right *int8, count int) int32 {
	if count == 0 {
		return 0
	}

	return DotInt8SSE2Asm(left, right, count)
}

//go:noescape
func DotInt8SSE2Asm(left, right *int8, count int) int32
