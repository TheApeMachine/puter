//go:build !cuda

package embedding

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (embedding *Embedding) Lookup(table, indices, output unsafe.Pointer, vocab, hidden, indexCount int, format dtype.DType,) {
	embedding.stubHost()
}
