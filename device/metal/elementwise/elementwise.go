package elementwise

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
BinaryKernel selects a binary elementwise Metal kernel.
*/
type BinaryKernel int

const (
	BinaryAdd BinaryKernel = iota
	BinarySub
	BinaryMul
	BinaryDiv
	BinaryMax
	BinaryMin
)

/*
UnaryKernel selects a unary elementwise Metal kernel.
*/
type UnaryKernel int

const (
	UnaryAbs UnaryKernel = iota
	UnaryNeg
	UnarySqrt
	UnaryReLU
)

/*
Elementwise implements device.Elementwise for the Metal backend.
Methods delegate kernel launch to a Host provided by the root Backend.
*/
type Elementwise struct {
	host Host
}

/*
New wires an Elementwise receiver to its Metal dispatch host.
*/
func New(host Host) Elementwise {
	return Elementwise{host: host}
}

/*
Host is the Metal dispatch surface elementwise operations call into.
*/
type Host interface {
	NeedsPlatform()
	BinaryElementwise(dst, left, right unsafe.Pointer, format dtype.DType, kernel BinaryKernel)
	UnaryElementwise(dst, src unsafe.Pointer, format dtype.DType, kernel UnaryKernel)
	DispatchAxpy(y, x unsafe.Pointer, alpha float32, format dtype.DType)
}
