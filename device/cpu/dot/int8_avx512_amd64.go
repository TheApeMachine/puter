//go:build amd64

package dot

//go:noescape
func DotInt8AVX512Asm(left, right *int8, count int) int32

func DotInt8AVX512(left, right *int8, count int) int32 {
	if count == 0 {
		return 0
	}

	return DotInt8AVX512Asm(left, right, count)
}
