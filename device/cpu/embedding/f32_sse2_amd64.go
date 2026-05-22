//go:build amd64

package embedding

//go:noescape
func CopyRowFloat32SSE2Asm(dst, src *float32, hidden int)

//go:noescape
func AddRowFloat32SSE2Asm(dst, src *float32, hidden int)

func copyRowF32SSE2(dst, src *float32, hidden int) {
	if hidden == 0 {
		return
	}

	CopyRowFloat32SSE2Asm(dst, src, hidden)
}

func addRowF32SSE2(dst, src *float32, hidden int) {
	if hidden == 0 {
		return
	}

	AddRowFloat32SSE2Asm(dst, src, hidden)
}
