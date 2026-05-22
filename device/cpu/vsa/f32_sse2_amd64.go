//go:build amd64

package vsa

//go:noescape
func VsaBindFloat32SSE2Asm(dst, left, right *float32, count int)

//go:noescape
func VsaBundleFloat32SSE2Asm(dst, left, right *float32, count int)

//go:noescape
func VsaPermuteCopyFloat32SSE2Asm(dst, src *float32, count int)

//go:noescape
func VsaSimilarityFloat32SSE2Asm(left, right *float32, count int) float32

func VsaBindF32SSE2(dst, left, right *float32, count int) {
	if count == 0 {
		return
	}

	VsaBindFloat32SSE2Asm(dst, left, right, count)
}

func VsaBundleF32SSE2(dst, left, right *float32, count int) {
	if count == 0 {
		return
	}

	VsaBundleFloat32SSE2Asm(dst, left, right, count)
}

func VsaPermuteCopyF32SSE2(dst, src *float32, count int) {
	if count == 0 {
		return
	}

	VsaPermuteCopyFloat32SSE2Asm(dst, src, count)
}

func VsaSimilarityF32SSE2(left, right *float32, count int) float32 {
	if count == 0 {
		return 0
	}

	return VsaSimilarityFloat32SSE2Asm(left, right, count)
}
