//go:build amd64

package dot

//go:noescape
func DotBFloat16AVX512Asm(left, right *uint16, count int) uint16

//go:noescape
func DotFloat16AVX512Asm(left, right *uint16, count int) uint16

func DotBF16AVX512(left, right *uint16, count int) uint16 {
	if count == 0 {
		return 0
	}

	return DotBFloat16AVX512Asm(left, right, count)
}

func DotFP16AVX512(left, right *uint16, count int) uint16 {
	if count == 0 {
		return 0
	}

	return DotFloat16AVX512Asm(left, right, count)
}
