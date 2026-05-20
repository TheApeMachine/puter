//go:build !arm64 && !amd64

package layernorm

func LayerNormSquaredDiffSumNative(row []float32, mean float32) float32 {
	return LayerNormSquaredDiffSumGeneric(row, mean)
}

func LayerNormApplyRowNative(
	outRow, row, scale, bias []float32,
	mean, invStdDev float32,
) {
	LayerNormApplyRowGeneric(outRow, row, scale, bias, mean, invStdDev)
}
