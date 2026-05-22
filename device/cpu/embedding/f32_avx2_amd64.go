//go:build amd64

package embedding

//go:noescape
func CopyRowFloat32AVX2Asm(dst, src *float32, hidden int)

//go:noescape
func AddRowFloat32AVX2Asm(dst, src *float32, hidden int)

func copyRowF32AVX2(dst, src *float32, hidden int) {
	if hidden == 0 {
		return
	}

	CopyRowFloat32AVX2Asm(dst, src, hidden)
}

func addRowF32AVX2(dst, src *float32, hidden int) {
	if hidden == 0 {
		return
	}

	AddRowFloat32AVX2Asm(dst, src, hidden)
}
