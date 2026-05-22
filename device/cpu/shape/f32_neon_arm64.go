//go:build arm64

package shape

//go:noescape
func CopyContiguousFloat32NEONAsm(dst, src *float32, count int)

//go:noescape
func WhereFloat32NEONAsm(dst, positive, negative *float32, mask *byte, count int)

//go:noescape
func MaskedFillFloat32NEONAsm(dst, input *float32, fill float32, mask *byte, count int)

func CopyContiguousF32NEON(dst, src *float32, count int) {
	if count == 0 {
		return
	}

	CopyContiguousFloat32NEONAsm(dst, src, count)
}

func WhereF32NEON(dst, positive, negative *float32, mask []byte, count int) {
	if count == 0 {
		return
	}

	WhereFloat32NEONAsm(dst, positive, negative, &mask[0], count)
}

func MaskedFillF32NEON(dst, input *float32, fill float32, mask []byte, count int) {
	if count == 0 {
		return
	}

	MaskedFillFloat32NEONAsm(dst, input, fill, &mask[0], count)
}
