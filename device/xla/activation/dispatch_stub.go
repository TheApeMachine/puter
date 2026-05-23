//go:build !xla

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

func (activation *Activation) Softmax(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) LogSoftmax(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) PReLU(dst, src unsafe.Pointer, count int, format dtype.DType, negativeSlope float32) {
	activation.stubHost()
}

func (activation *Activation) PReLUV(dst, src, slopes unsafe.Pointer, count int, format dtype.DType, slopeCount int) {
	activation.stubHost()
}

func (activation *Activation) LeakyReLUSlope(dst, src unsafe.Pointer, count int, format dtype.DType, negativeSlope float32) {
	activation.stubHost()
}

func (activation *Activation) ELUAlpha(dst, src unsafe.Pointer, count int, format dtype.DType, alpha float32) {
	activation.stubHost()
}

func (activation *Activation) CELUAlpha(dst, src unsafe.Pointer, count int, format dtype.DType, alpha float32) {
	activation.stubHost()
}

func (activation *Activation) Threshold(dst, src unsafe.Pointer, count int, format dtype.DType, threshold float32) {
	activation.stubHost()
}

func (activation *Activation) HardTanhRange(dst, src unsafe.Pointer, count int, format dtype.DType, minVal, maxVal float32) {
	activation.stubHost()
}

func (activation *Activation) Snake(dst, src unsafe.Pointer, count int, format dtype.DType, alpha float32) {
	activation.stubHost()
}

func (activation *Activation) SnakeParametric(dst, src unsafe.Pointer, count int, format dtype.DType, alpha, beta float32) {
	activation.stubHost()
}

func (activation *Activation) HardShrink(dst, src unsafe.Pointer, count int, format dtype.DType, lambda float32) {
	activation.stubHost()
}

func (activation *Activation) SoftShrink(dst, src unsafe.Pointer, count int, format dtype.DType, lambda float32) {
	activation.stubHost()
}

func (activation *Activation) RReLU(dst, src unsafe.Pointer, count int, format dtype.DType, lower, upper float32) {
	activation.stubHost()
}

func (activation *Activation) GLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) GeGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) GeGLUTanh(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) SwiGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) ReGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) SiGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) GLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) GeGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) GeGLUTanhTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) SwiGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) ReGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) SiGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) LinGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) SeGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) LinGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) SeGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}

