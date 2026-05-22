//go:build arm64

package tokenizer

import "testing"

func BenchmarkTokenizerPackInt32NEON(b *testing.B) {
	const benchLen = 8192

	source := randomInt32Vector(benchLen, 0x2840)
	destination := make([]int32, benchLen)

	b.SetBytes(int64(benchLen * 4))
	b.ResetTimer()

	for b.Loop() {
		TokenizerPackInt32NEON(&destination[0], &source[0], benchLen)
	}
}
