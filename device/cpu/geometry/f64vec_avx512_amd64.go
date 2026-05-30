//go:build amd64

package geometry

//go:noescape
func SumFloat64AVX512Asm(src *float64, count int) float64

//go:noescape
func SumOfSquaresFloat64AVX512Asm(src *float64, count int) float64

//go:noescape
func DotFloat64AVX512Asm(left, right *float64, count int) float64

//go:noescape
func MaxFloat64AVX512Asm(src *float64, count int) float64

//go:noescape
func ScaleFloat64AVX512Asm(dst, src *float64, scale float64, count int)

//go:noescape
func AddScalarFloat64AVX512Asm(dst, src *float64, offset float64, count int)

//go:noescape
func MulFloat64AVX512Asm(dst, left, right *float64, count int)

//go:noescape
func AddFloat64AVX512Asm(dst, left, right *float64, count int)

//go:noescape
func SqrtFloat64AVX512Asm(dst, src *float64, count int)

func sumFloat64AVX512(values []float64) float64 {
	return SumFloat64AVX512Asm(&values[0], len(values))
}

func sumOfSquaresFloat64AVX512(values []float64) float64 {
	return SumOfSquaresFloat64AVX512Asm(&values[0], len(values))
}

func dotFloat64AVX512(left, right []float64) float64 {
	return DotFloat64AVX512Asm(&left[0], &right[0], len(left))
}

func maxFloat64AVX512(values []float64) float64 {
	return MaxFloat64AVX512Asm(&values[0], len(values))
}

func scaleFloat64AVX512(destination, source []float64, scale float64) {
	ScaleFloat64AVX512Asm(&destination[0], &source[0], scale, len(destination))
}

func addScalarFloat64AVX512(destination, source []float64, offset float64) {
	AddScalarFloat64AVX512Asm(&destination[0], &source[0], offset, len(destination))
}

func mulFloat64AVX512(destination, left, right []float64) {
	MulFloat64AVX512Asm(&destination[0], &left[0], &right[0], len(destination))
}

func addFloat64AVX512(destination, left, right []float64) {
	AddFloat64AVX512Asm(&destination[0], &left[0], &right[0], len(destination))
}

func sqrtFloat64AVX512(destination, source []float64) {
	SqrtFloat64AVX512Asm(&destination[0], &source[0], len(destination))
}
