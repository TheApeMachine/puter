package sampling

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

/*
SamplingKernel selects a Metal sampling kernel.
*/
type SamplingKernel int

const (
	KernelGreedy SamplingKernel = iota
	KernelTopK
	KernelTopP
)

/*
Sampling implements device.Sampling for the Metal backend.
Methods delegate kernel launch to a Host provided by the root Backend.
*/
type Sampling struct {
	host Host
}

/*
New wires a Sampling receiver to its Metal dispatch host.
*/
func New(host Host) Sampling {
	return Sampling{host: host}
}

/*
Host is the Metal dispatch surface sampling operations call into.
*/
type Host interface {
	NeedsPlatform()
	SamplingIndex(
		dst unsafe.Pointer,
		kernel SamplingKernel,
		config device.SamplingConfig,
		logits unsafe.Pointer,
		vocabSize int,
		format dtype.DType,
	)
}
