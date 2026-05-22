//go:build amd64

package vsa

import "golang.org/x/sys/cpu"

func simdCopyBlockNative(dst, src []float32) {
	elementCount := len(src)

	if cpu.X86.HasAVX512F {
		VsaPermuteCopyF32AVX512(&dst[0], &src[0], elementCount)

		return
	}

	if cpu.X86.HasAVX2 && cpu.X86.HasFMA {
		VsaPermuteCopyF32AVX2(&dst[0], &src[0], elementCount)

		return
	}

	if cpu.X86.HasSSE2 {
		VsaPermuteCopyF32SSE2(&dst[0], &src[0], elementCount)

		return
	}

	for index := range src {
		dst[index] = src[index]
	}
}

func simdBindNative(dst, left, right []float32) {
	elementCount := len(dst)

	if cpu.X86.HasAVX512F {
		VsaBindF32AVX512(&dst[0], &left[0], &right[0], elementCount)

		return
	}

	if cpu.X86.HasAVX2 && cpu.X86.HasFMA {
		VsaBindF32AVX2(&dst[0], &left[0], &right[0], elementCount)

		return
	}

	if cpu.X86.HasSSE2 {
		VsaBindF32SSE2(&dst[0], &left[0], &right[0], elementCount)

		return
	}

	VsaBindFloat32Scalar(dst, left, right)
}

func simdBundleNative(dst, left, right []float32) {
	elementCount := len(dst)

	if cpu.X86.HasAVX512F {
		VsaBundleF32AVX512(&dst[0], &left[0], &right[0], elementCount)

		return
	}

	if cpu.X86.HasAVX2 && cpu.X86.HasFMA {
		VsaBundleF32AVX2(&dst[0], &left[0], &right[0], elementCount)

		return
	}

	if cpu.X86.HasSSE2 {
		VsaBundleF32SSE2(&dst[0], &left[0], &right[0], elementCount)

		return
	}

	VsaBundleFloat32Scalar(dst, left, right)
}

func simdSimilarityNative(left, right []float32) float32 {
	elementCount := len(left)

	if cpu.X86.HasAVX512F {
		return VsaSimilarityF32AVX512(&left[0], &right[0], elementCount)
	}

	if cpu.X86.HasAVX2 && cpu.X86.HasFMA {
		return VsaSimilarityF32AVX2(&left[0], &right[0], elementCount)
	}

	if cpu.X86.HasSSE2 {
		return VsaSimilarityF32SSE2(&left[0], &right[0], elementCount)
	}

	return VsaSimilarityFloat32Scalar(left, right)
}
