//go:build amd64

package normalization

import "golang.org/x/sys/cpu"

func NormSquaredDiffSumNative(row []float32, mean float32) float32 {
	if cpu.X86.HasAVX512F {
		return normSquaredDiffSumF32AVX512(row, mean)
	}

	if cpu.X86.HasAVX2 && cpu.X86.HasFMA {
		return normSquaredDiffSumF32AVX2(row, mean)
	}

	if cpu.X86.HasSSE2 {
		return normSquaredDiffSumF32SSE2(row, mean)
	}

	return NormSquaredDiffSumGeneric(row, mean)
}

func NormApplyConstScaleBiasNative(
	outRow, row []float32,
	mean, invStdDev, scale, bias float32,
) {
	if cpu.X86.HasAVX512F {
		normApplyConstScaleBiasF32AVX512(outRow, row, mean, invStdDev, scale, bias)

		return
	}

	if cpu.X86.HasAVX2 && cpu.X86.HasFMA {
		normApplyConstScaleBiasF32AVX2(outRow, row, mean, invStdDev, scale, bias)

		return
	}

	if cpu.X86.HasSSE2 {
		normApplyConstScaleBiasF32SSE2(outRow, row, mean, invStdDev, scale, bias)

		return
	}

	NormApplyConstScaleBiasGeneric(outRow, row, mean, invStdDev, scale, bias)
}
