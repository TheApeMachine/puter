//go:build xla

package embedding

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

func (embedding *Embedding) Lookup(
	table, indices, output unsafe.Pointer,
	vocab, hidden, indexCount int,
	format dtype.DType,
) {
	embedding.host.DispatchEmbeddingLookup(table, indices, output, vocab, hidden, indexCount, format)
}

func (embedding *Embedding) Bag(
	table, indices, offsets, output unsafe.Pointer,
	vocab, hidden, bagCount, indexCount int,
	format dtype.DType,
) {
	embedding.host.DispatchEmbeddingBag(
		table, indices, offsets, output,
		vocab, hidden, bagCount, indexCount,
		format,
	)
}

func (embedding *Embedding) TimestepEmbedding(
	config device.TimestepEmbeddingConfig,
	timesteps, output unsafe.Pointer,
	count, dim int,
	format dtype.DType,
) {
	embedding.host.DispatchTimestepEmbedding(config, timesteps, output, count, dim, format)
}
