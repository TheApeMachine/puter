//go:build amd64

package layernorm

import "golang.org/x/sys/cpu"

func LayerNormSquaredDiffSumNative(row []float32, mean float32) float32 {
	if cpu.X86.HasAVX512F {
		return layerNormSquaredDiffSumF32AVX512(row, mean)
	}

	if cpu.X86.HasAVX2 && cpu.X86.HasFMA {
		return layerNormSquaredDiffSumF32AVX2(row, mean)
	}

	if cpu.X86.HasSSE2 {
		return layerNormSquaredDiffSumF32SSE2(row, mean)
	}

	return LayerNormSquaredDiffSumGeneric(row, mean)
}

func LayerNormApplyRowNative(
	outRow, row, scale, bias []float32,
	mean, invStdDev float32,
) {
	if cpu.X86.HasAVX512F {
		layerNormApplyRowF32AVX512(outRow, row, scale, bias, mean, invStdDev)

		return
	}

	if cpu.X86.HasAVX2 && cpu.X86.HasFMA {
		layerNormApplyRowF32AVX2(outRow, row, scale, bias, mean, invStdDev)

		return
	}

	if cpu.X86.HasSSE2 {
		layerNormApplyRowF32SSE2(outRow, row, scale, bias, mean, invStdDev)

		return
	}

	LayerNormApplyRowGeneric(outRow, row, scale, bias, mean, invStdDev)
}
