//go:build arm64

package normalization

func NormSquaredDiffSumNative(row []float32, mean float32) float32 {
	return normSquaredDiffSumF32NEON(row, mean)
}

func NormApplyConstScaleBiasNative(
	outRow, row []float32,
	mean, invStdDev, scale, bias float32,
) {
	normApplyConstScaleBiasF32NEON(outRow, row, mean, invStdDev, scale, bias)
}
