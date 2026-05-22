//go:build arm64

package elementwise

//go:noescape
func AxpyBFloat16NEONAsm(y, x *uint16, alpha float32, count int)

//go:noescape
func AxpyFloat16NEONAsm(y, x *uint16, alpha float32, count int)

func AxpyBF16NEON(y, x *uint16, alpha float32, count int) {
	if count == 0 {
		return
	}

	AxpyBFloat16NEONAsm(y, x, alpha, count)
}

func AxpyF16NEON(y, x *uint16, alpha float32, count int) {
	if count == 0 {
		return
	}

	AxpyFloat16NEONAsm(y, x, alpha, count)
}
