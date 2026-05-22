//go:build amd64

package vsa

//go:noescape
func VsaBindFloat32AVX2Asm(dst, left, right *float32, count int)

//go:noescape
func VsaBundleFloat32AVX2Asm(dst, left, right *float32, count int)

//go:noescape
func VsaPermuteCopyFloat32AVX2Asm(dst, src *float32, count int)

//go:noescape
func VsaSimilarityFloat32AVX2Asm(left, right *float32, count int) float32

func VsaBindF32AVX2(dst, left, right *float32, count int) {
	if count == 0 {
		return
	}

	VsaBindFloat32AVX2Asm(dst, left, right, count)
}

func VsaBundleF32AVX2(dst, left, right *float32, count int) {
	if count == 0 {
		return
	}

	VsaBundleFloat32AVX2Asm(dst, left, right, count)
}

func VsaPermuteCopyF32AVX2(dst, src *float32, count int) {
	if count == 0 {
		return
	}

	VsaPermuteCopyFloat32AVX2Asm(dst, src, count)
}

func VsaSimilarityF32AVX2(left, right *float32, count int) float32 {
	if count == 0 {
		return 0
	}

	return VsaSimilarityFloat32AVX2Asm(left, right, count)
}
