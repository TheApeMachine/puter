//go:build arm64

package shape

import "testing"

func BenchmarkCopyContiguousF32NEON(b *testing.B) {
	length := 8192
	source := randomShapeFloat32(length, 0x1870)
	destination := make([]float32, length)

	b.SetBytes(int64(length * 4))
	b.ResetTimer()

	for b.Loop() {
		CopyContiguousF32NEON(&destination[0], &source[0], length)
	}
}

func BenchmarkWhereF32NEON(b *testing.B) {
	length := 8192
	positive := randomShapeFloat32(length, 0x1871)
	negative := randomShapeFloat32(length, 0x1872)
	mask := shapeMaskBytes(length, 0x1873)
	destination := make([]float32, length)

	b.SetBytes(int64(length * 4))
	b.ResetTimer()

	for b.Loop() {
		WhereF32NEON(&destination[0], &positive[0], &negative[0], mask, length)
	}
}

func BenchmarkMaskedFillF32NEON(b *testing.B) {
	length := 8192
	input := randomShapeFloat32(length, 0x1874)
	mask := shapeMaskBytes(length, 0x1875)
	destination := make([]float32, length)

	b.SetBytes(int64(length * 4))
	b.ResetTimer()

	for b.Loop() {
		MaskedFillF32NEON(&destination[0], &input[0], 1.5, mask, length)
	}
}
