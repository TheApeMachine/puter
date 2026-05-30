//go:build arm64

package geometry

//go:noescape
func SumFloat64NEONAsm(src *float64, count int) float64

//go:noescape
func SumOfSquaresFloat64NEONAsm(src *float64, count int) float64

//go:noescape
func DotFloat64NEONAsm(left, right *float64, count int) float64

//go:noescape
func MaxFloat64NEONAsm(src *float64, count int) float64

//go:noescape
func ScaleFloat64NEONAsm(dst, src *float64, scale float64, count int)

//go:noescape
func AddScalarFloat64NEONAsm(dst, src *float64, offset float64, count int)

//go:noescape
func MulFloat64NEONAsm(dst, left, right *float64, count int)

//go:noescape
func AddFloat64NEONAsm(dst, left, right *float64, count int)

//go:noescape
func SqrtFloat64NEONAsm(dst, src *float64, count int)

func sumFloat64NEON(values []float64) float64 {
	return SumFloat64NEONAsm(&values[0], len(values))
}

func sumOfSquaresFloat64NEON(values []float64) float64 {
	return SumOfSquaresFloat64NEONAsm(&values[0], len(values))
}

func dotFloat64NEON(left, right []float64) float64 {
	return DotFloat64NEONAsm(&left[0], &right[0], len(left))
}

func maxFloat64NEON(values []float64) float64 {
	return MaxFloat64NEONAsm(&values[0], len(values))
}

func scaleFloat64NEON(destination, source []float64, scale float64) {
	ScaleFloat64NEONAsm(&destination[0], &source[0], scale, len(destination))
}

func addScalarFloat64NEON(destination, source []float64, offset float64) {
	AddScalarFloat64NEONAsm(&destination[0], &source[0], offset, len(destination))
}

func mulFloat64NEON(destination, left, right []float64) {
	MulFloat64NEONAsm(&destination[0], &left[0], &right[0], len(destination))
}

func addFloat64NEON(destination, left, right []float64) {
	AddFloat64NEONAsm(&destination[0], &left[0], &right[0], len(destination))
}

func sqrtFloat64NEON(destination, source []float64) {
	SqrtFloat64NEONAsm(&destination[0], &source[0], len(destination))
}
