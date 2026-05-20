package cpu

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cpu/activation"
)

func (backend *Backend) Exp(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.Exp(dst, src, count, format)
}

func (backend *Backend) Log(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.Log(dst, src, count, format)
}

func (backend *Backend) Log1p(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.Log1p(dst, src, count, format)
}

func (backend *Backend) Expm1(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.Expm1(dst, src, count, format)
}

func (backend *Backend) Sigmoid(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.Sigmoid(dst, src, count, format)
}

func (backend *Backend) LogSigmoid(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.LogSigmoid(dst, src, count, format)
}

func (backend *Backend) Tanh(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.Tanh(dst, src, count, format)
}

func (backend *Backend) Silu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.Silu(dst, src, count, format)
}

func (backend *Backend) Swish(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.Swish(dst, src, count, format)
}

func (backend *Backend) GeluTanh(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.GeluTanh(dst, src, count, format)
}

func (backend *Backend) Gelu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.Gelu(dst, src, count, format)
}

func (backend *Backend) LeakyReLU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.LeakyReLU(dst, src, count, format)
}

func (backend *Backend) ELU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.ELU(dst, src, count, format)
}

func (backend *Backend) CELU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.CELU(dst, src, count, format)
}

func (backend *Backend) SELU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.SELU(dst, src, count, format)
}

func (backend *Backend) Softplus(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.Softplus(dst, src, count, format)
}

func (backend *Backend) Mish(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.Mish(dst, src, count, format)
}

func (backend *Backend) Softsign(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.Softsign(dst, src, count, format)
}

func (backend *Backend) HardSigmoid(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.HardSigmoid(dst, src, count, format)
}

func (backend *Backend) HardSwish(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.HardSwish(dst, src, count, format)
}

func (backend *Backend) HardTanh(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.HardTanh(dst, src, count, format)
}

func (backend *Backend) HardGelu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.HardGelu(dst, src, count, format)
}

func (backend *Backend) QuickGelu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.QuickGelu(dst, src, count, format)
}

func (backend *Backend) TanhShrink(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.TanhShrink(dst, src, count, format)
}

func (backend *Backend) Softmax(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.Softmax(dst, src, count, format)
}

func (backend *Backend) LogSoftmax(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.LogSoftmax(dst, src, count, format)
}

func (backend *Backend) PReLU(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	negativeSlope float32,
) {
	activation.PReLU(dst, src, count, format, negativeSlope)
}

func (backend *Backend) PReLUV(
	dst, src, slopes unsafe.Pointer,
	count int,
	format dtype.DType,
	slopeCount int,
) {
	activation.PReLUV(dst, src, slopes, count, format, slopeCount)
}

func (backend *Backend) LeakyReLUSlope(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	negativeSlope float32,
) {
	activation.LeakyReLUSlope(dst, src, count, format, negativeSlope)
}

func (backend *Backend) ELUAlpha(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	alpha float32,
) {
	activation.ELUAlpha(dst, src, count, format, alpha)
}

func (backend *Backend) CELUAlpha(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	alpha float32,
) {
	activation.CELUAlpha(dst, src, count, format, alpha)
}

func (backend *Backend) Threshold(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	threshold float32,
) {
	activation.Threshold(dst, src, count, format, threshold)
}

func (backend *Backend) HardTanhRange(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	minVal, maxVal float32,
) {
	activation.HardTanhRange(dst, src, count, format, minVal, maxVal)
}

func (backend *Backend) Snake(dst, src unsafe.Pointer, count int, format dtype.DType, alpha float32) {
	activation.Snake(dst, src, count, format, alpha)
}

func (backend *Backend) SnakeParametric(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	alpha, beta float32,
) {
	activation.SnakeParametric(dst, src, count, format, alpha, beta)
}

func (backend *Backend) HardShrink(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	lambda float32,
) {
	activation.HardShrink(dst, src, count, format, lambda)
}

func (backend *Backend) SoftShrink(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	lambda float32,
) {
	activation.SoftShrink(dst, src, count, format, lambda)
}

func (backend *Backend) RReLU(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	lower, upper float32,
) {
	activation.RReLU(dst, src, count, format, lower, upper)
}

func (backend *Backend) GLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	activation.GLU(dst, packed, batch, halfCount, format)
}

func (backend *Backend) GeGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	activation.GeGLU(dst, packed, batch, halfCount, format)
}

func (backend *Backend) GeGLUTanh(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	activation.GeGLUTanh(dst, packed, batch, halfCount, format)
}

func (backend *Backend) SwiGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	activation.SwiGLU(dst, packed, batch, halfCount, format)
}

func (backend *Backend) ReGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	activation.ReGLU(dst, packed, batch, halfCount, format)
}

func (backend *Backend) SiGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	activation.SiGLU(dst, packed, batch, halfCount, format)
}

func (backend *Backend) GLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	activation.GLUTensors(dst, gate, up, count, format)
}

func (backend *Backend) GeGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	activation.GeGLUTensors(dst, gate, up, count, format)
}

func (backend *Backend) GeGLUTanhTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	activation.GeGLUTanhTensors(dst, gate, up, count, format)
}

func (backend *Backend) SwiGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	activation.SwiGLUTensors(dst, gate, up, count, format)
}

func (backend *Backend) ReGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	activation.ReGLUTensors(dst, gate, up, count, format)
}

func (backend *Backend) SiGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	activation.SiGLUTensors(dst, gate, up, count, format)
}

func (backend *Backend) LinGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	activation.LinGLU(dst, packed, batch, halfCount, format)
}

func (backend *Backend) SeGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	activation.SeGLU(dst, packed, batch, halfCount, format)
}

func (backend *Backend) LinGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	activation.LinGLUTensors(dst, gate, up, count, format)
}

func (backend *Backend) SeGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	activation.SeGLUTensors(dst, gate, up, count, format)
}
