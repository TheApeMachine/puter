package reduction

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
Reduction implements device.Reduction for the XLA backend.
*/
type Reduction struct {
	host Host
}

/*
Host is the XLA dispatch surface reduction operations call into.
*/
type Host interface {
	NeedsPlatform()
	NotImplemented(methodName string)
	ReductionScalar(
		dst unsafe.Pointer,
		values unsafe.Pointer,
		count int,
		format dtype.DType,
		kernel ReductionKernel,
	)
}

/*
ReductionKernel selects an XLA reduction program.
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
New wires a Reduction receiver to its XLA dispatch host.
*/
func New(host Host) Reduction {
	return Reduction{host: host}
}

func (reduction *Reduction) stubHost() {
	reduction.host.NeedsPlatform()
}

func (reduction *Reduction) unimplemented(methodName string) {
	reduction.host.NotImplemented(methodName)
}
