package embedding

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

/*
Embedding implements device.Embedding for the Metal backend.
Methods delegate kernel launch to a Host provided by the root Backend.
*/
type Embedding struct {
	host Host
}

/*
New wires an Embedding receiver to its Metal dispatch host.
*/
func New(host Host) Embedding {
	return Embedding{host: host}
}

/*
Host is the Metal dispatch surface embedding operations call into.
*/
type Host interface {
	NeedsPlatform()
	LaunchLookup(
		table, indices, output unsafe.Pointer,
		vocab, hidden, indexCount int,
		format dtype.DType,
	)
	LaunchBag(
		table, indices, offsets, output unsafe.Pointer,
		vocab, hidden, bagCount, indexCount int,
		format dtype.DType,
	)
	LaunchTimestepEmbedding(
		config device.TimestepEmbeddingConfig,
		timesteps, output unsafe.Pointer,
		count, dim int,
		format dtype.DType,
	)
}
