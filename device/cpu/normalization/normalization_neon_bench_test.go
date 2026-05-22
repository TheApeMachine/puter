//go:build arm64

package normalization

import "testing"

func BenchmarkNormSquaredDiffSumF32NEON(b *testing.B) {
	row := randomNormalizationRow(8192, 1)
	mean := float32(0.1)
	length := len(row)

	b.SetBytes(int64(length * 4))
	b.ResetTimer()

	for b.Loop() {
		_ = normSquaredDiffSumF32NEON(row, mean)
	}
}

func BenchmarkNormApplyConstScaleBiasF32NEON(b *testing.B) {
	row := randomNormalizationRow(8192, 2)
	out := make([]float32, len(row))
	length := len(row)

	b.SetBytes(int64(length * 8))
	b.ResetTimer()

	for b.Loop() {
		normApplyConstScaleBiasF32NEON(out, row, 0.05, 1.25, 0.9, -0.1)
	}
}
