//go:build arm64

package embedding

//go:noescape
func CopyRowFloat32NEONAsm(dst, src *float32, hidden int)

//go:noescape
func AddRowFloat32NEONAsm(dst, src *float32, hidden int)

func copyRowF32NEON(dst, src *float32, hidden int) {
	if hidden == 0 {
		return
	}

	CopyRowFloat32NEONAsm(dst, src, hidden)
}

func addRowF32NEON(dst, src *float32, hidden int) {
	if hidden == 0 {
		return
	}

	AddRowFloat32NEONAsm(dst, src, hidden)
}
