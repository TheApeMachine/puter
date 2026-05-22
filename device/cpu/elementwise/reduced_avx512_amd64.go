//go:build amd64

package elementwise

//go:noescape
func AddBFloat16AVX512Asm(dst, left, right *uint16, count int)

//go:noescape
func SubBFloat16AVX512Asm(dst, left, right *uint16, count int)

//go:noescape
func MulBFloat16AVX512Asm(dst, left, right *uint16, count int)

//go:noescape
func DivBFloat16AVX512Asm(dst, left, right *uint16, count int)

//go:noescape
func MaxBFloat16AVX512Asm(dst, left, right *uint16, count int)

//go:noescape
func MinBFloat16AVX512Asm(dst, left, right *uint16, count int)

//go:noescape
func AbsBFloat16AVX512Asm(dst, src *uint16, count int)

//go:noescape
func NegBFloat16AVX512Asm(dst, src *uint16, count int)

//go:noescape
func SqrtBFloat16AVX512Asm(dst, src *uint16, count int)

//go:noescape
func ReluBFloat16AVX512Asm(dst, src *uint16, count int)

//go:noescape
func AddFloat16AVX512Asm(dst, left, right *uint16, count int)

//go:noescape
func SubFloat16AVX512Asm(dst, left, right *uint16, count int)

//go:noescape
func MulFloat16AVX512Asm(dst, left, right *uint16, count int)

//go:noescape
func DivFloat16AVX512Asm(dst, left, right *uint16, count int)

//go:noescape
func MaxFloat16AVX512Asm(dst, left, right *uint16, count int)

//go:noescape
func MinFloat16AVX512Asm(dst, left, right *uint16, count int)

//go:noescape
func AbsFloat16AVX512Asm(dst, src *uint16, count int)

//go:noescape
func NegFloat16AVX512Asm(dst, src *uint16, count int)

//go:noescape
func SqrtFloat16AVX512Asm(dst, src *uint16, count int)

//go:noescape
func ReluFloat16AVX512Asm(dst, src *uint16, count int)

//go:noescape
func AxpyBFloat16AVX512Asm(y, x *uint16, alpha float32, count int)

//go:noescape
func AxpyFloat16AVX512Asm(y, x *uint16, alpha float32, count int)

func AddBF16AVX512(dst, left, right *uint16, count int) {
	if count == 0 {
		return
	}

	AddBFloat16AVX512Asm(dst, left, right, count)
}

func SubBF16AVX512(dst, left, right *uint16, count int) {
	if count == 0 {
		return
	}

	SubBFloat16AVX512Asm(dst, left, right, count)
}

func MulBF16AVX512(dst, left, right *uint16, count int) {
	if count == 0 {
		return
	}

	MulBFloat16AVX512Asm(dst, left, right, count)
}

func DivBF16AVX512(dst, left, right *uint16, count int) {
	if count == 0 {
		return
	}

	DivBFloat16AVX512Asm(dst, left, right, count)
}

func MaxBF16AVX512(dst, left, right *uint16, count int) {
	if count == 0 {
		return
	}

	MaxBFloat16AVX512Asm(dst, left, right, count)
}

func MinBF16AVX512(dst, left, right *uint16, count int) {
	if count == 0 {
		return
	}

	MinBFloat16AVX512Asm(dst, left, right, count)
}

func AbsBF16AVX512(dst, src *uint16, count int) {
	if count == 0 {
		return
	}

	AbsBFloat16AVX512Asm(dst, src, count)
}

func NegBF16AVX512(dst, src *uint16, count int) {
	if count == 0 {
		return
	}

	NegBFloat16AVX512Asm(dst, src, count)
}

func SqrtBF16AVX512(dst, src *uint16, count int) {
	if count == 0 {
		return
	}

	SqrtBFloat16AVX512Asm(dst, src, count)
}

func ReluBF16AVX512(dst, src *uint16, count int) {
	if count == 0 {
		return
	}

	ReluBFloat16AVX512Asm(dst, src, count)
}

func AddF16AVX512(dst, left, right *uint16, count int) {
	if count == 0 {
		return
	}

	AddFloat16AVX512Asm(dst, left, right, count)
}

func SubF16AVX512(dst, left, right *uint16, count int) {
	if count == 0 {
		return
	}

	SubFloat16AVX512Asm(dst, left, right, count)
}

func MulF16AVX512(dst, left, right *uint16, count int) {
	if count == 0 {
		return
	}

	MulFloat16AVX512Asm(dst, left, right, count)
}

func DivF16AVX512(dst, left, right *uint16, count int) {
	if count == 0 {
		return
	}

	DivFloat16AVX512Asm(dst, left, right, count)
}

func MaxF16AVX512(dst, left, right *uint16, count int) {
	if count == 0 {
		return
	}

	MaxFloat16AVX512Asm(dst, left, right, count)
}

func MinF16AVX512(dst, left, right *uint16, count int) {
	if count == 0 {
		return
	}

	MinFloat16AVX512Asm(dst, left, right, count)
}

func AbsF16AVX512(dst, src *uint16, count int) {
	if count == 0 {
		return
	}

	AbsFloat16AVX512Asm(dst, src, count)
}

func NegF16AVX512(dst, src *uint16, count int) {
	if count == 0 {
		return
	}

	NegFloat16AVX512Asm(dst, src, count)
}

func SqrtF16AVX512(dst, src *uint16, count int) {
	if count == 0 {
		return
	}

	SqrtFloat16AVX512Asm(dst, src, count)
}

func ReluF16AVX512(dst, src *uint16, count int) {
	if count == 0 {
		return
	}

	ReluFloat16AVX512Asm(dst, src, count)
}

func AxpyBF16AVX512(y, x *uint16, alpha float32, count int) {
	if count == 0 {
		return
	}

	AxpyBFloat16AVX512Asm(y, x, alpha, count)
}

func AxpyF16AVX512(y, x *uint16, alpha float32, count int) {
	if count == 0 {
		return
	}

	AxpyFloat16AVX512Asm(y, x, alpha, count)
}
