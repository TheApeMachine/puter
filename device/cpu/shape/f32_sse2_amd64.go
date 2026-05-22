//go:build amd64

package shape

//go:noescape
func CopyContiguousFloat32SSE2Asm(dst, src *float32, count int)

//go:noescape
func WhereFloat32SSE2Asm(dst, positive, negative *float32, mask *byte, count int)

//go:noescape
func MaskedFillFloat32SSE2Asm(dst, input *float32, fill float32, mask *byte, count int)

func CopyContiguousF32SSE2(dst, src *float32, count int) {
	if count == 0 {
		return
	}

	CopyContiguousFloat32SSE2Asm(dst, src, count)
}

func WhereF32SSE2(dst, positive, negative *float32, mask []byte, count int) {
	if count == 0 {
		return
	}

	WhereFloat32SSE2Asm(dst, positive, negative, &mask[0], count)
}

func MaskedFillF32SSE2(dst, input *float32, fill float32, mask []byte, count int) {
	if count == 0 {
		return
	}

	MaskedFillFloat32SSE2Asm(dst, input, fill, &mask[0], count)
}
