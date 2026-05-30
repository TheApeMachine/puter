//go:build amd64

package geometry

//go:noescape
func SumFloat64SSE2Asm(src *float64, count int) float64

//go:noescape
func SumOfSquaresFloat64SSE2Asm(src *float64, count int) float64

//go:noescape
func DotFloat64SSE2Asm(left, right *float64, count int) float64

//go:noescape
func MaxFloat64SSE2Asm(src *float64, count int) float64

//go:noescape
func ScaleFloat64SSE2Asm(dst, src *float64, scale float64, count int)

//go:noescape
func AddScalarFloat64SSE2Asm(dst, src *float64, offset float64, count int)

//go:noescape
func MulFloat64SSE2Asm(dst, left, right *float64, count int)

//go:noescape
func AddFloat64SSE2Asm(dst, left, right *float64, count int)

//go:noescape
func SqrtFloat64SSE2Asm(dst, src *float64, count int)

func sumFloat64SSE2(values []float64) float64 {
	return SumFloat64SSE2Asm(&values[0], len(values))
}

func sumOfSquaresFloat64SSE2(values []float64) float64 {
	return SumOfSquaresFloat64SSE2Asm(&values[0], len(values))
}

func dotFloat64SSE2(left, right []float64) float64 {
	return DotFloat64SSE2Asm(&left[0], &right[0], len(left))
}

func maxFloat64SSE2(values []float64) float64 {
	return MaxFloat64SSE2Asm(&values[0], len(values))
}

func scaleFloat64SSE2(destination, source []float64, scale float64) {
	ScaleFloat64SSE2Asm(&destination[0], &source[0], scale, len(destination))
}

func addScalarFloat64SSE2(destination, source []float64, offset float64) {
	AddScalarFloat64SSE2Asm(&destination[0], &source[0], offset, len(destination))
}

func mulFloat64SSE2(destination, left, right []float64) {
	MulFloat64SSE2Asm(&destination[0], &left[0], &right[0], len(destination))
}

func addFloat64SSE2(destination, left, right []float64) {
	AddFloat64SSE2Asm(&destination[0], &left[0], &right[0], len(destination))
}

func sqrtFloat64SSE2(destination, source []float64) {
	SqrtFloat64SSE2Asm(&destination[0], &source[0], len(destination))
}
