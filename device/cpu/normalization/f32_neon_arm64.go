//go:build arm64

package normalization

//go:noescape
func NormSquaredDiffSumFloat32NEONAsm(row *float32, count int, mean float32) float32

//go:noescape
func NormApplyConstScaleBiasFloat32NEONAsm(
	out, row *float32,
	count int,
	mean, invStdDev, scale, bias float32,
)

func normSquaredDiffSumF32NEON(row []float32, mean float32) float32 {
	if len(row) == 0 {
		return 0
	}

	return NormSquaredDiffSumFloat32NEONAsm(&row[0], len(row), mean)
}

func normApplyConstScaleBiasF32NEON(
	outRow, row []float32,
	mean, invStdDev, scale, bias float32,
) {
	elementCount := len(row)

	if elementCount == 0 {
		return
	}

	NormApplyConstScaleBiasFloat32NEONAsm(
		&outRow[0], &row[0], elementCount,
		mean, invStdDev, scale, bias,
	)
}
