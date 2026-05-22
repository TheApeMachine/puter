//go:build arm64

package model_editing

import "testing"

func BenchmarkWeightGraftAddFloat32NEON(b *testing.B) {
	const benchLen = 8192

	weights, injection := randomGraftVectors(benchLen, 0x2B61)

	b.SetBytes(int64(benchLen * 4))
	b.ResetTimer()

	for b.Loop() {
		WeightGraftAddFloat32NEON(&weights[0], &injection[0], benchLen)
	}
}
