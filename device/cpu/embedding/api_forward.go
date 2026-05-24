package embedding

import (
	"unsafe"
	"github.com/theapemachine/manifesto/dtype"
)

var defaultEmbedding = New()

func Bag(table, indices, offsets, output unsafe.Pointer,
	vocab, hidden, bagCount, indexCount int,
	format dtype.DType) {
	defaultEmbedding.Bag(table, indices, offsets, output, vocab, hidden, bagCount, indexCount, format)
}

func Lookup(table, indices, output unsafe.Pointer,
	vocab, hidden, indexCount int,
	format dtype.DType) {
	defaultEmbedding.Lookup(table, indices, output, vocab, hidden, indexCount, format)
}
