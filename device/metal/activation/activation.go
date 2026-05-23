package activation

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
StandardKernel selects a standard unary activation Metal kernel.
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
GLUVariant selects a gated linear unit Metal kernel.
*/
type GLUVariant int

const (
	GLU GLUVariant = iota
	GeGLU
	GeGLUTanh
	SwiGLU
	ReGLU
	SiGLU
	LinGLU
	SeGLU
)

/*
Activation implements device.Activation for the Metal backend.
Methods delegate kernel launch to a Host provided by the root Backend.
*/
type Activation struct {
	host Host
}

/*
New wires an Activation receiver to its Metal dispatch host.
*/
func New(host Host) Activation {
	return Activation{host: host}
}

/*
Host is the Metal dispatch surface activation operations call into.
*/
type Host interface {
	NeedsPlatform()
	StandardUnary(dst, src unsafe.Pointer, format dtype.DType, kernel StandardKernel)
	Softmax(dst, src unsafe.Pointer, format dtype.DType)
	UnaryParam(dst, src unsafe.Pointer, format dtype.DType, kernelName string, param float32)
	PReLUV(dst, src, slopes unsafe.Pointer, format dtype.DType)
	GLUPacked(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType, variant GLUVariant)
	GLUTensors(dst, gate, up unsafe.Pointer, format dtype.DType, variant GLUVariant)
}
