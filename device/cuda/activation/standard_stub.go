//go:build !cuda

package activation

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (activation *Activation) Exp(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) Log(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) Log1p(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) Expm1(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) Sigmoid(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) LogSigmoid(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) Tanh(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) Silu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) Swish(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) GeluTanh(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) Gelu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) ReLU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) LeakyReLU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) ELU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) CELU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) SELU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) Softplus(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) Mish(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) Softsign(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) HardSigmoid(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) HardSwish(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) HardTanh(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) HardGelu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) QuickGelu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) TanhShrink(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}
