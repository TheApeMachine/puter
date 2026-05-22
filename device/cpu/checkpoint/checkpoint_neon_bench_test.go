//go:build arm64

package checkpoint

import "testing"

func BenchmarkCheckpointEncodeFloat32DataNEON(b *testing.B) {
	const benchLen = 8192

	source := randomFloat32Vector(benchLen, 0x2980)
	destination := make([]uint8, benchLen*4)

	b.SetBytes(int64(benchLen * 4))
	b.ResetTimer()

	for b.Loop() {
		CheckpointEncodeFloat32DataNEON(&destination[0], &source[0], benchLen)
	}
}

func BenchmarkCheckpointDecodeFloat32DataNEON(b *testing.B) {
	const benchLen = 8192

	source := randomFloat32Vector(benchLen, 0x2990)
	payload := make([]uint8, benchLen*4)
	encodeFloat32DataScalar(payload, source)
	destination := make([]float32, benchLen)

	b.SetBytes(int64(benchLen * 4))
	b.ResetTimer()

	for b.Loop() {
		CheckpointDecodeFloat32DataNEON(&destination[0], &payload[0], benchLen)
	}
}
