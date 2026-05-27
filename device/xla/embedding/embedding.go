package embedding

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

/*
Embedding implements device.Embedding for the XLA backend.
*/
type Embedding struct {
	host Host
}

/*
Host is the XLA dispatch surface embedding operations call into.
*/
type Host interface {
	NeedsPlatform()
	DispatchEmbeddingLookup(
		table, indices, output unsafe.Pointer,
		vocab, hidden, indexCount int,
		format dtype.DType,
	)
	DispatchEmbeddingBag(
		table, indices, offsets, output unsafe.Pointer,
		vocab, hidden, bagCount, indexCount int,
		format dtype.DType,
	)
	DispatchTimestepEmbedding(
		config device.TimestepEmbeddingConfig,
		timesteps, output unsafe.Pointer,
		count, dim int,
		format dtype.DType,
	)
	NotImplemented(string)
}

/*
New wires a Embedding receiver to its XLA dispatch host.
*/
func New(host Host) Embedding {
	return Embedding{host: host}
}

func (receiver *Embedding) stubHost() {
	receiver.host.NeedsPlatform()
}

func (receiver *Embedding) unimplemented(methodName string) {
	receiver.host.NotImplemented(methodName)
}
