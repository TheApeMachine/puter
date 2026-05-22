//go:build arm64

package math

import "testing"

func BenchmarkInvSqrtDimScaleF32NEON(b *testing.B) {
	input := randomMathFloat32(8192, 0x3220)
	output := make([]float32, 8192)

	b.ResetTimer()

	for b.Loop() {
		InvSqrtDimScaleF32NEON(output, input, 64)
	}
}

func BenchmarkInvSqrtDimScaleFloat32NEONAsm(b *testing.B) {
	input := randomMathFloat32(8192, 0x3221)
	output := make([]float32, 8192)
	scale := float32(1.0 / 8.0)

	b.ResetTimer()

	for b.Loop() {
		InvSqrtDimScaleFloat32NEONAsm(&output[0], &input[0], scale, len(output))
	}
}

func BenchmarkLogSumExpRowF32NEON(b *testing.B) {
	row := randomMathFloat32(8192, 0x3222)

	b.ResetTimer()

	for b.Loop() {
		_ = logSumExpRowF32NEON(row)
	}
}

func BenchmarkOuterF32NEON(b *testing.B) {
	left := randomMathFloat32(64, 0x3223)
	right := randomMathFloat32(128, 0x3224)
	output := make([]float32, len(left)*len(right))

	b.ResetTimer()

	for b.Loop() {
		OuterF32NEON(left, right, output)
	}
}

func BenchmarkInvSqrtDimScaleGeneric(b *testing.B) {
	input := randomMathFloat32(8192, 0x3225)
	output := make([]float32, 8192)

	b.ResetTimer()

	for b.Loop() {
		InvSqrtDimScaleGeneric(output, input, 64)
	}
}
