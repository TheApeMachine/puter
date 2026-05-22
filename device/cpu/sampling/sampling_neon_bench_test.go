//go:build arm64

package sampling

import "testing"

func BenchmarkGreedySampleF32NEON(b *testing.B) {
	logits := randomSamplingLogits(8192, 0x3620)

	b.ResetTimer()

	for b.Loop() {
		_ = GreedySampleF32NEON(&logits[0], len(logits))
	}
}

func BenchmarkGreedySampleFloat32NEONAsm(b *testing.B) {
	logits := randomSamplingLogits(8192, 0x3621)

	b.ResetTimer()

	for b.Loop() {
		_ = GreedySampleFloat32NEONAsm(&logits[0], len(logits))
	}
}

func BenchmarkSamplingSoftmaxRowF32NEON(b *testing.B) {
	logits := randomSamplingLogits(8192, 0x3622)
	output := make([]float32, 8192)
	temperature := float32(0.85)

	b.ResetTimer()

	for b.Loop() {
		SamplingSoftmaxRowF32NEON(&logits[0], &output[0], temperature, len(logits))
	}
}

func BenchmarkSamplingSoftmaxRowGeneric(b *testing.B) {
	logits := randomSamplingLogits(8192, 0x3623)
	output := make([]float32, 8192)
	temperature := float32(0.85)

	b.ResetTimer()

	for b.Loop() {
		SamplingSoftmaxRowGeneric(logits, output, temperature)
	}
}
