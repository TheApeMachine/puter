package elementwise

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
UnaryKernel selects a unary elementwise XLA lowering.
*/
type UnaryKernel int

const (
	UnaryAbs UnaryKernel = iota
	UnaryNeg
	UnarySqrt
	UnaryReLU
)

/*
BinaryKernel selects a binary elementwise XLA lowering.
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
Elementwise implements device.Elementwise for the XLA backend.
*/
type Elementwise struct {
	host Host
}

/*
Host is the XLA dispatch surface elementwise operations call into.
*/
type Host interface {
	NeedsPlatform()
	NotImplemented(methodName string)
	UnaryElementwise(dst, src unsafe.Pointer, format dtype.DType, kernel UnaryKernel)
	BinaryElementwise(dst, left, right unsafe.Pointer, format dtype.DType, kernel BinaryKernel)
}

/*
New wires an Elementwise receiver to its XLA dispatch host.
*/
func New(host Host) Elementwise {
	return Elementwise{host: host}
}

func (elementwise *Elementwise) stubHost() {
	elementwise.host.NeedsPlatform()
}

func (elementwise *Elementwise) unimplemented(methodName string) {
	elementwise.host.NotImplemented(methodName)
}

/*
UnaryKernelName maps unary elementwise kernels to XLA operation names.
*/
func UnaryKernelName(kernel UnaryKernel) (string, bool) {
	switch kernel {
	case UnaryAbs:
		return "abs", true
	case UnaryNeg:
		return "neg", true
	case UnarySqrt:
		return "sqrt", true
	case UnaryReLU:
		return "relu", true
	default:
		return "", false
	}
}

/*
BinaryKernelName maps binary elementwise kernels to XLA operation names.
*/
func BinaryKernelName(kernel BinaryKernel) (string, bool) {
	switch kernel {
	case BinaryAdd:
		return "add", true
	case BinarySub:
		return "sub", true
	case BinaryMul:
		return "mul", true
	case BinaryDiv:
		return "div", true
	case BinaryMax:
		return "max", true
	case BinaryMin:
		return "min", true
	default:
		return "", false
	}
}
