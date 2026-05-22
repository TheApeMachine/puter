//go:build amd64

package vsa

func VsaBindFloat32Native(out, left, right []float32) {
	if len(out) == 0 {
		return
	}

	simdBindNative(out, left, right)
}

func VsaBundleFloat32Native(out, left, right []float32) {
	if len(out) == 0 {
		return
	}

	simdBundleNative(out, left, right)
}

func VsaPermuteFloat32Native(out, in []float32, shift int) {
	elementCount := len(in)

	if elementCount == 0 {
		return
	}

	if shift == 0 {
		copyBlockNative(out, in)

		return
	}

	firstLength := elementCount - shift
	secondLength := shift

	if firstLength > 0 {
		copyBlockNative(out[:firstLength], in[shift:shift+firstLength])
	}

	if secondLength > 0 {
		copyBlockNative(out[firstLength:], in[:secondLength])
	}
}

func VsaSimilarityFloat32Native(left, right []float32) float32 {
	if len(left) == 0 {
		return 0
	}

	return simdSimilarityNative(left, right)
}

func copyBlockNative(dst, src []float32) {
	simdCopyBlockNative(dst, src)
}
