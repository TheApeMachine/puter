//go:build amd64

package elementwise

func AddBF16AVX2(dst, left, right *uint16, count int) {
	AddBF16AVX512(dst, left, right, count)
}

func SubBF16AVX2(dst, left, right *uint16, count int) {
	SubBF16AVX512(dst, left, right, count)
}

func MulBF16AVX2(dst, left, right *uint16, count int) {
	MulBF16AVX512(dst, left, right, count)
}

func DivBF16AVX2(dst, left, right *uint16, count int) {
	DivBF16AVX512(dst, left, right, count)
}

func MaxBF16AVX2(dst, left, right *uint16, count int) {
	MaxBF16AVX512(dst, left, right, count)
}

func MinBF16AVX2(dst, left, right *uint16, count int) {
	MinBF16AVX512(dst, left, right, count)
}

func AbsBF16AVX2(dst, src *uint16, count int) {
	AbsBF16AVX512(dst, src, count)
}

func NegBF16AVX2(dst, src *uint16, count int) {
	NegBF16AVX512(dst, src, count)
}

func SqrtBF16AVX2(dst, src *uint16, count int) {
	SqrtBF16AVX512(dst, src, count)
}

func ReluBF16AVX2(dst, src *uint16, count int) {
	ReluBF16AVX512(dst, src, count)
}

func AxpyBF16AVX2(y, x *uint16, alpha float32, count int) {
	AxpyBF16AVX512(y, x, alpha, count)
}

func AddF16AVX2(dst, left, right *uint16, count int) {
	AddF16AVX512(dst, left, right, count)
}

func SubF16AVX2(dst, left, right *uint16, count int) {
	SubF16AVX512(dst, left, right, count)
}

func MulF16AVX2(dst, left, right *uint16, count int) {
	MulF16AVX512(dst, left, right, count)
}

func DivF16AVX2(dst, left, right *uint16, count int) {
	DivF16AVX512(dst, left, right, count)
}

func MaxF16AVX2(dst, left, right *uint16, count int) {
	MaxF16AVX512(dst, left, right, count)
}

func MinF16AVX2(dst, left, right *uint16, count int) {
	MinF16AVX512(dst, left, right, count)
}

func AbsF16AVX2(dst, src *uint16, count int) {
	AbsF16AVX512(dst, src, count)
}

func NegF16AVX2(dst, src *uint16, count int) {
	NegF16AVX512(dst, src, count)
}

func SqrtF16AVX2(dst, src *uint16, count int) {
	SqrtF16AVX512(dst, src, count)
}

func ReluF16AVX2(dst, src *uint16, count int) {
	ReluF16AVX512(dst, src, count)
}

func AxpyF16AVX2(y, x *uint16, alpha float32, count int) {
	AxpyF16AVX512(y, x, alpha, count)
}
