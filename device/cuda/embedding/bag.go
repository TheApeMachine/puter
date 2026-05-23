//go:build cuda

package embedding

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (embedding *Embedding) Bag(
	table, indices, offsets, output unsafe.Pointer,
	vocab, hidden, bagCount, indexCount int,
	format dtype.DType,
) {
	embedding.host.LaunchBag(table, indices, offsets, output, vocab, hidden, bagCount, indexCount, format)
}
