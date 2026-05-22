//go:build arm64

package embedding

import (
	"testing"
	"unsafe"
)

func BenchmarkLookupF32NEON(b *testing.B) {
	const vocab = 8192
	const hidden = 8192
	const indexCount = 64

	table := randomEmbeddingTable(vocab, hidden, 0xE240)
	indices := randomEmbeddingIndices(indexCount, vocab, 0xE241)
	output := make([]float32, indexCount*hidden)

	b.SetBytes(int64(indexCount * hidden * 4))
	b.ResetTimer()

	for b.Loop() {
		runLookupF32NEON(
			unsafe.Pointer(&table[0]),
			unsafe.Pointer(&indices[0]),
			unsafe.Pointer(&output[0]),
			vocab, hidden, indexCount,
		)
	}
}

func BenchmarkCopyRowF32NEON(b *testing.B) {
	const hidden = 8192

	table := randomEmbeddingTable(2, hidden, 0xE244)
	dst := make([]float32, hidden)

	b.SetBytes(int64(hidden * 4))
	b.ResetTimer()

	for b.Loop() {
		copyRowF32NEON(&dst[0], &table[hidden], hidden)
	}
}

func BenchmarkAddRowF32NEON(b *testing.B) {
	const hidden = 8192

	table := randomEmbeddingTable(2, hidden, 0xE245)
	dst := make([]float32, hidden)

	for index := range dst {
		dst[index] = float32(index) * 0.01
	}

	b.SetBytes(int64(hidden * 4))
	b.ResetTimer()

	for b.Loop() {
		addRowF32NEON(&dst[0], &table[hidden], hidden)
	}
}
