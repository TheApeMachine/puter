package matmul

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
Gemm implements device.Matmul for the XLA backend.
*/
type Gemm struct {
	host Host
}

/*
Host is the XLA dispatch surface matmul operations call into.
*/
type Host interface {
	NeedsPlatform()
	NotImplemented(methodName string)
	MatmulLaunch(
		out, left, right unsafe.Pointer,
		rows, inner, cols int,
		format dtype.DType,
	)
}

/*
New wires a Gemm receiver to its XLA dispatch host.
*/
func New(host Host) Gemm {
	return Gemm{host: host}
}

func (gemm *Gemm) stubHost() {
	gemm.host.NeedsPlatform()
}

func (gemm *Gemm) unimplemented(methodName string) {
	gemm.host.NotImplemented(methodName)
}
