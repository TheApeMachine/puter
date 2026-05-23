package matmul

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
Gemm implements device.Matmul for the Metal backend.
*/
type Gemm struct {
	host Host
}

/*
New wires a Gemm receiver to its Metal dispatch host.
*/
func New(host Host) Gemm {
	return Gemm{host: host}
}

/*
Host is the Metal dispatch surface matmul operations call into.
*/
type Host interface {
	NeedsPlatform()
	MatmulLaunch(
		out, left, right unsafe.Pointer,
		rows, inner, cols int,
		format dtype.DType,
	)
}
