//go:build arm64

package masking

import "testing"

func BenchmarkApplyMaskF32NEON(b *testing.B) {
	length := 8192
	input := randomMaskingFloat32(length, 0x1940)
	mask := randomMaskingFloat32(length, 0x1941)
	output := make([]float32, length)

	b.SetBytes(int64(length * 4))
	b.ResetTimer()

	for b.Loop() {
		ApplyMaskF32NEON(&input[0], &mask[0], &output[0], length)
	}
}

func BenchmarkCausalMaskF32NEON(b *testing.B) {
	side := 128
	output := make([]float32, side*side)

	b.SetBytes(int64(side * side * 4))
	b.ResetTimer()

	for b.Loop() {
		CausalMaskF32NEON(&output[0], side, side)
	}
}

func BenchmarkALiBiBiasF32NEON(b *testing.B) {
	side := 128
	scores := randomMaskingScores(side, side, 0x1942)
	slope := []float32{0.125}
	output := make([]float32, side*side)

	b.SetBytes(int64(side * side * 4))
	b.ResetTimer()

	for b.Loop() {
		ALiBiBiasF32NEON(&scores[0], &slope[0], &output[0], side, side)
	}
}
