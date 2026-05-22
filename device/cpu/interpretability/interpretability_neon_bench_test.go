//go:build arm64

package interpretability

import "testing"

func BenchmarkActivationSteerFloat32NEON(b *testing.B) {
	const benchLen = 8192
	const coefficient = float32(0.25)

	base, direction := randomSteerVectors(benchLen, 0x2A61)
	destination := make([]float32, benchLen)

	b.SetBytes(int64(benchLen * 4))
	b.ResetTimer()

	for b.Loop() {
		ActivationSteerFloat32NEON(
			&destination[0], &base[0], &direction[0], coefficient, benchLen,
		)
	}
}
