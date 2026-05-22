//go:build amd64

package elementwise

//go:noescape
func AddBFloat16SSE2Asm(dst, left, right *uint16, count int)

//go:noescape
func SubBFloat16SSE2Asm(dst, left, right *uint16, count int)

//go:noescape
func MulBFloat16SSE2Asm(dst, left, right *uint16, count int)

//go:noescape
func DivBFloat16SSE2Asm(dst, left, right *uint16, count int)

//go:noescape
func MaxBFloat16SSE2Asm(dst, left, right *uint16, count int)

//go:noescape
func MinBFloat16SSE2Asm(dst, left, right *uint16, count int)

//go:noescape
func AbsBFloat16SSE2Asm(dst, src *uint16, count int)

//go:noescape
func NegBFloat16SSE2Asm(dst, src *uint16, count int)

//go:noescape
func SqrtBFloat16SSE2Asm(dst, src *uint16, count int)

//go:noescape
func ReluBFloat16SSE2Asm(dst, src *uint16, count int)

//go:noescape
func AxpyBFloat16SSE2Asm(y, x *uint16, alpha float32, count int)

//go:noescape
func AddFloat16SSE2Asm(dst, left, right *uint16, count int)

//go:noescape
func SubFloat16SSE2Asm(dst, left, right *uint16, count int)

//go:noescape
func MulFloat16SSE2Asm(dst, left, right *uint16, count int)

//go:noescape
func DivFloat16SSE2Asm(dst, left, right *uint16, count int)

//go:noescape
func MaxFloat16SSE2Asm(dst, left, right *uint16, count int)

//go:noescape
func MinFloat16SSE2Asm(dst, left, right *uint16, count int)

//go:noescape
func AbsFloat16SSE2Asm(dst, src *uint16, count int)

//go:noescape
func NegFloat16SSE2Asm(dst, src *uint16, count int)

//go:noescape
func SqrtFloat16SSE2Asm(dst, src *uint16, count int)

//go:noescape
func ReluFloat16SSE2Asm(dst, src *uint16, count int)

//go:noescape
func AxpyFloat16SSE2Asm(y, x *uint16, alpha float32, count int)

func AddBF16SSE2(dst, left, right *uint16, count int) {
	if count == 0 {
		return
	}

	AddBFloat16SSE2Asm(dst, left, right, count)
}

func SubBF16SSE2(dst, left, right *uint16, count int) {
	if count == 0 {
		return
	}

	SubBFloat16SSE2Asm(dst, left, right, count)
}

func MulBF16SSE2(dst, left, right *uint16, count int) {
	if count == 0 {
		return
	}

	MulBFloat16SSE2Asm(dst, left, right, count)
}

func DivBF16SSE2(dst, left, right *uint16, count int) {
	if count == 0 {
		return
	}

	DivBFloat16SSE2Asm(dst, left, right, count)
}

func MaxBF16SSE2(dst, left, right *uint16, count int) {
	if count == 0 {
		return
	}

	MaxBFloat16SSE2Asm(dst, left, right, count)
}

func MinBF16SSE2(dst, left, right *uint16, count int) {
	if count == 0 {
		return
	}

	MinBFloat16SSE2Asm(dst, left, right, count)
}

func AbsBF16SSE2(dst, src *uint16, count int) {
	if count == 0 {
		return
	}

	AbsBFloat16SSE2Asm(dst, src, count)
}

func NegBF16SSE2(dst, src *uint16, count int) {
	if count == 0 {
		return
	}

	NegBFloat16SSE2Asm(dst, src, count)
}

func SqrtBF16SSE2(dst, src *uint16, count int) {
	if count == 0 {
		return
	}

	SqrtBFloat16SSE2Asm(dst, src, count)
}

func ReluBF16SSE2(dst, src *uint16, count int) {
	if count == 0 {
		return
	}

	ReluBFloat16SSE2Asm(dst, src, count)
}

func AxpyBF16SSE2(y, x *uint16, alpha float32, count int) {
	if count == 0 {
		return
	}

	AxpyBFloat16SSE2Asm(y, x, alpha, count)
}

func AddF16SSE2(dst, left, right *uint16, count int) {
	if count == 0 {
		return
	}

	AddFloat16SSE2Asm(dst, left, right, count)
}

func SubF16SSE2(dst, left, right *uint16, count int) {
	if count == 0 {
		return
	}

	SubFloat16SSE2Asm(dst, left, right, count)
}

func MulF16SSE2(dst, left, right *uint16, count int) {
	if count == 0 {
		return
	}

	MulFloat16SSE2Asm(dst, left, right, count)
}

func DivF16SSE2(dst, left, right *uint16, count int) {
	if count == 0 {
		return
	}

	DivFloat16SSE2Asm(dst, left, right, count)
}

func MaxF16SSE2(dst, left, right *uint16, count int) {
	if count == 0 {
		return
	}

	MaxFloat16SSE2Asm(dst, left, right, count)
}

func MinF16SSE2(dst, left, right *uint16, count int) {
	if count == 0 {
		return
	}

	MinFloat16SSE2Asm(dst, left, right, count)
}

func AbsF16SSE2(dst, src *uint16, count int) {
	if count == 0 {
		return
	}

	AbsFloat16SSE2Asm(dst, src, count)
}

func NegF16SSE2(dst, src *uint16, count int) {
	if count == 0 {
		return
	}

	NegFloat16SSE2Asm(dst, src, count)
}

func SqrtF16SSE2(dst, src *uint16, count int) {
	if count == 0 {
		return
	}

	SqrtFloat16SSE2Asm(dst, src, count)
}

func ReluF16SSE2(dst, src *uint16, count int) {
	if count == 0 {
		return
	}

	ReluFloat16SSE2Asm(dst, src, count)
}

func AxpyF16SSE2(y, x *uint16, alpha float32, count int) {
	if count == 0 {
		return
	}

	AxpyFloat16SSE2Asm(y, x, alpha, count)
}
