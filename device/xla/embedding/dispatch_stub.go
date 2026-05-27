//go:build !xla

package embedding

import (
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
	"unsafe"
)

func (embedding *Embedding) Lookup(table, indices, output unsafe.Pointer, vocab, hidden, indexCount int, format dtype.DType) {
	embedding.stubHost()
}

func (embedding *Embedding) Bag(table, indices, offsets, output unsafe.Pointer, vocab, hidden, bagCount, indexCount int, format dtype.DType) {
	embedding.stubHost()
}

func (embedding *Embedding) TimestepEmbedding(config device.TimestepEmbeddingConfig, timesteps, output unsafe.Pointer, count, dim int, format dtype.DType) {
	embedding.stubHost()
}
