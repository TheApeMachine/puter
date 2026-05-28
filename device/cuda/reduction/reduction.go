package reduction

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
ReductionKernel selects a Metal reduction kernel.
*/
type ReductionKernel int

const (
	KernelSum ReductionKernel = iota
	KernelProd
	KernelMin
	KernelMax
	KernelL1Norm
)

/*
Reduction implements device.Reduction for the Metal backend.
Methods delegate kernel launch to a Host provided by the root Backend.
*/
type Reduction struct {
	host Host
}

/*
New wires a Reduction receiver to its Metal dispatch host.
*/
func New(host Host) Reduction {
	return Reduction{host: host}
}

/*
Host is the Metal dispatch surface reduction operations call into.
*/
type Host interface {
	NeedsPlatform()
	ReductionScalar(
		dst unsafe.Pointer,
		values unsafe.Pointer,
		count int,
		format dtype.DType,
		kernel ReductionKernel,
	)
}
