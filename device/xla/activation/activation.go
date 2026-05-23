package activation

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
StandardKernel selects a standard unary activation XLA lowering.
*/
type StandardKernel int

const (
	StandardExp StandardKernel = iota
	StandardLog
	StandardLog1p
	StandardExpm1
	StandardSigmoid
	StandardLogSigmoid
	StandardTanh
	StandardSilu
	StandardSwish
	StandardGeluTanh
	StandardGelu
	StandardReLU
	StandardLeakyReLU
	StandardELU
	StandardCELU
	StandardSELU
	StandardSoftplus
	StandardMish
	StandardSoftsign
	StandardHardSigmoid
	StandardHardSwish
	StandardHardTanh
	StandardHardGelu
	StandardQuickGelu
	StandardTanhShrink
)

/*
Activation implements device.Activation for the XLA backend.
*/
type Activation struct {
	host Host
}

/*
Host is the XLA dispatch surface activation operations call into.
*/
type Host interface {
	NeedsPlatform()
	NotImplemented(methodName string)
	StandardUnary(dst, src unsafe.Pointer, format dtype.DType, kernel StandardKernel)
	UnaryParam(dst, src unsafe.Pointer, format dtype.DType, kernelName string, param float32)
}

/*
New wires an Activation receiver to its XLA dispatch host.
*/
func New(host Host) Activation {
	return Activation{host: host}
}

func (activation *Activation) stubHost() {
	activation.host.NeedsPlatform()
}

func (activation *Activation) unimplemented(methodName string) {
	activation.host.NotImplemented(methodName)
}
