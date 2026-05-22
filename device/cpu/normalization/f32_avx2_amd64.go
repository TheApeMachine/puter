//go:build amd64

package normalization

//go:noescape
func NormSquaredDiffSumFloat32AVX2Asm(row *float32, count int, mean float32) float32

//go:noescape
func NormApplyConstScaleBiasFloat32AVX2Asm(
	out, row *float32,
	count int,
	mean, invStdDev, scale, bias float32,
)

func normSquaredDiffSumF32AVX2(row []float32, mean float32) float32 {
	if len(row) == 0 {
		return 0
	}

	return NormSquaredDiffSumFloat32AVX2Asm(&row[0], len(row), mean)
}

func normApplyConstScaleBiasF32AVX2(
	outRow, row []float32,
	mean, invStdDev, scale, bias float32,
) {
	elementCount := len(row)

	if elementCount == 0 {
		return
	}

	NormApplyConstScaleBiasFloat32AVX2Asm(
		&outRow[0], &row[0], elementCount,
		mean, invStdDev, scale, bias,
	)
}
