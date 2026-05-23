//go:build cuda

package activation

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (activation *Activation) Exp(dst, src unsafe.Pointer, count int, format dtype.DType) {
	_ = count
	activation.host.StandardUnary(dst, src, format, StandardExp)
}

func (activation *Activation) Log(dst, src unsafe.Pointer, count int, format dtype.DType) {
	_ = count
	activation.host.StandardUnary(dst, src, format, StandardLog)
}

func (activation *Activation) Log1p(dst, src unsafe.Pointer, count int, format dtype.DType) {
	_ = count
	activation.host.StandardUnary(dst, src, format, StandardLog1p)
}

func (activation *Activation) Expm1(dst, src unsafe.Pointer, count int, format dtype.DType) {
	_ = count
	activation.host.StandardUnary(dst, src, format, StandardExpm1)
}

func (activation *Activation) Sigmoid(dst, src unsafe.Pointer, count int, format dtype.DType) {
	_ = count
	activation.host.StandardUnary(dst, src, format, StandardSigmoid)
}

func (activation *Activation) LogSigmoid(dst, src unsafe.Pointer, count int, format dtype.DType) {
	_ = count
	activation.host.StandardUnary(dst, src, format, StandardLogSigmoid)
}

func (activation *Activation) Tanh(dst, src unsafe.Pointer, count int, format dtype.DType) {
	_ = count
	activation.host.StandardUnary(dst, src, format, StandardTanh)
}

func (activation *Activation) Silu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	_ = count
	activation.host.StandardUnary(dst, src, format, StandardSilu)
}

func (activation *Activation) Swish(dst, src unsafe.Pointer, count int, format dtype.DType) {
	_ = count
	activation.host.StandardUnary(dst, src, format, StandardSwish)
}

func (activation *Activation) GeluTanh(dst, src unsafe.Pointer, count int, format dtype.DType) {
	_ = count
	activation.host.StandardUnary(dst, src, format, StandardGeluTanh)
}

func (activation *Activation) Gelu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	_ = count
	activation.host.StandardUnary(dst, src, format, StandardGelu)
}

func (activation *Activation) ReLU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	_ = count
	activation.host.StandardUnary(dst, src, format, StandardReLU)
}

func (activation *Activation) LeakyReLU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	_ = count
	activation.host.StandardUnary(dst, src, format, StandardLeakyReLU)
}

func (activation *Activation) ELU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	_ = count
	activation.host.StandardUnary(dst, src, format, StandardELU)
}

func (activation *Activation) CELU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	_ = count
	activation.host.StandardUnary(dst, src, format, StandardCELU)
}

func (activation *Activation) SELU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	_ = count
	activation.host.StandardUnary(dst, src, format, StandardSELU)
}

func (activation *Activation) Softplus(dst, src unsafe.Pointer, count int, format dtype.DType) {
	_ = count
	activation.host.StandardUnary(dst, src, format, StandardSoftplus)
}

func (activation *Activation) Mish(dst, src unsafe.Pointer, count int, format dtype.DType) {
	_ = count
	activation.host.StandardUnary(dst, src, format, StandardMish)
}

func (activation *Activation) Softsign(dst, src unsafe.Pointer, count int, format dtype.DType) {
	_ = count
	activation.host.StandardUnary(dst, src, format, StandardSoftsign)
}

func (activation *Activation) HardSigmoid(dst, src unsafe.Pointer, count int, format dtype.DType) {
	_ = count
	activation.host.StandardUnary(dst, src, format, StandardHardSigmoid)
}

func (activation *Activation) HardSwish(dst, src unsafe.Pointer, count int, format dtype.DType) {
	_ = count
	activation.host.StandardUnary(dst, src, format, StandardHardSwish)
}

func (activation *Activation) HardTanh(dst, src unsafe.Pointer, count int, format dtype.DType) {
	_ = count
	activation.host.StandardUnary(dst, src, format, StandardHardTanh)
}

func (activation *Activation) HardGelu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	_ = count
	activation.host.StandardUnary(dst, src, format, StandardHardGelu)
}

func (activation *Activation) QuickGelu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	_ = count
	activation.host.StandardUnary(dst, src, format, StandardQuickGelu)
}

func (activation *Activation) TanhShrink(dst, src unsafe.Pointer, count int, format dtype.DType) {
	_ = count
	activation.host.StandardUnary(dst, src, format, StandardTanhShrink)
}
