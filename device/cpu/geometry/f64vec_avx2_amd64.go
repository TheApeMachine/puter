//go:build amd64

package geometry

//go:noescape
func SumFloat64AVX2Asm(src *float64, count int) float64

//go:noescape
func SumOfSquaresFloat64AVX2Asm(src *float64, count int) float64

//go:noescape
func DotFloat64AVX2Asm(left, right *float64, count int) float64

//go:noescape
func MaxFloat64AVX2Asm(src *float64, count int) float64

//go:noescape
func ScaleFloat64AVX2Asm(dst, src *float64, scale float64, count int)

//go:noescape
func AddScalarFloat64AVX2Asm(dst, src *float64, offset float64, count int)

//go:noescape
func MulFloat64AVX2Asm(dst, left, right *float64, count int)

//go:noescape
func AddFloat64AVX2Asm(dst, left, right *float64, count int)

//go:noescape
func SqrtFloat64AVX2Asm(dst, src *float64, count int)

func sumFloat64AVX2(values []float64) float64 {
	return SumFloat64AVX2Asm(&values[0], len(values))
}

func sumOfSquaresFloat64AVX2(values []float64) float64 {
	return SumOfSquaresFloat64AVX2Asm(&values[0], len(values))
}

func dotFloat64AVX2(left, right []float64) float64 {
	return DotFloat64AVX2Asm(&left[0], &right[0], len(left))
}

func maxFloat64AVX2(values []float64) float64 {
	return MaxFloat64AVX2Asm(&values[0], len(values))
}

func scaleFloat64AVX2(destination, source []float64, scale float64) {
	ScaleFloat64AVX2Asm(&destination[0], &source[0], scale, len(destination))
}

func addScalarFloat64AVX2(destination, source []float64, offset float64) {
	AddScalarFloat64AVX2Asm(&destination[0], &source[0], offset, len(destination))
}

func mulFloat64AVX2(destination, left, right []float64) {
	MulFloat64AVX2Asm(&destination[0], &left[0], &right[0], len(destination))
}

func addFloat64AVX2(destination, left, right []float64) {
	AddFloat64AVX2Asm(&destination[0], &left[0], &right[0], len(destination))
}

func sqrtFloat64AVX2(destination, source []float64) {
	SqrtFloat64AVX2Asm(&destination[0], &source[0], len(destination))
}
