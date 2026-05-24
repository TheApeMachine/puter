package activation

import (
	"unsafe"
	"github.com/theapemachine/manifesto/dtype"
)

var defaultActivation = New()

func CELU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	defaultActivation.CELU(dst, src, count, format)
}

func CELUAlpha(dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	alpha float32) {
	defaultActivation.CELUAlpha(dst, src, count, format, alpha)
}

func ELU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	defaultActivation.ELU(dst, src, count, format)
}

func ELUAlpha(dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	alpha float32) {
	defaultActivation.ELUAlpha(dst, src, count, format, alpha)
}

func Exp(dst, src unsafe.Pointer, count int, format dtype.DType) {
	defaultActivation.Exp(dst, src, count, format)
}

func Expm1(dst, src unsafe.Pointer, count int, format dtype.DType) {
	defaultActivation.Expm1(dst, src, count, format)
}

func GLU(dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType) {
	defaultActivation.GLU(dst, packed, batch, halfCount, format)
}

func GLUTensors(dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType) {
	defaultActivation.GLUTensors(dst, gate, up, count, format)
}

func GeGLU(dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType) {
	defaultActivation.GeGLU(dst, packed, batch, halfCount, format)
}

func GeGLUTanh(dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType) {
	defaultActivation.GeGLUTanh(dst, packed, batch, halfCount, format)
}

func GeGLUTanhTensors(dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType) {
	defaultActivation.GeGLUTanhTensors(dst, gate, up, count, format)
}

func GeGLUTensors(dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType) {
	defaultActivation.GeGLUTensors(dst, gate, up, count, format)
}

func Gelu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	defaultActivation.Gelu(dst, src, count, format)
}

func GeluTanh(dst, src unsafe.Pointer, count int, format dtype.DType) {
	defaultActivation.GeluTanh(dst, src, count, format)
}

func HardGelu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	defaultActivation.HardGelu(dst, src, count, format)
}

func HardShrink(dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	lambda float32) {
	defaultActivation.HardShrink(dst, src, count, format, lambda)
}

func HardSigmoid(dst, src unsafe.Pointer, count int, format dtype.DType) {
	defaultActivation.HardSigmoid(dst, src, count, format)
}

func HardSwish(dst, src unsafe.Pointer, count int, format dtype.DType) {
	defaultActivation.HardSwish(dst, src, count, format)
}

func HardTanh(dst, src unsafe.Pointer, count int, format dtype.DType) {
	defaultActivation.HardTanh(dst, src, count, format)
}

func HardTanhRange(dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	minVal, maxVal float32) {
	defaultActivation.HardTanhRange(dst, src, count, format, minVal, maxVal)
}

func LeakyReLU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	defaultActivation.LeakyReLU(dst, src, count, format)
}

func LeakyReLUSlope(dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	negativeSlope float32) {
	defaultActivation.LeakyReLUSlope(dst, src, count, format, negativeSlope)
}

func LinGLU(dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType) {
	defaultActivation.LinGLU(dst, packed, batch, halfCount, format)
}

func LinGLUTensors(dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType) {
	defaultActivation.LinGLUTensors(dst, gate, up, count, format)
}

func Log(dst, src unsafe.Pointer, count int, format dtype.DType) {
	defaultActivation.Log(dst, src, count, format)
}

func Log1p(dst, src unsafe.Pointer, count int, format dtype.DType) {
	defaultActivation.Log1p(dst, src, count, format)
}

func LogSigmoid(dst, src unsafe.Pointer, count int, format dtype.DType) {
	defaultActivation.LogSigmoid(dst, src, count, format)
}

func LogSoftmax(dst, src unsafe.Pointer, count int, format dtype.DType) {
	defaultActivation.LogSoftmax(dst, src, count, format)
}

func Mish(dst, src unsafe.Pointer, count int, format dtype.DType) {
	defaultActivation.Mish(dst, src, count, format)
}

func PReLU(dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	negativeSlope float32) {
	defaultActivation.PReLU(dst, src, count, format, negativeSlope)
}

func PReLUV(dst, src, slopes unsafe.Pointer,
	count int,
	format dtype.DType,
	slopeCount int) {
	defaultActivation.PReLUV(dst, src, slopes, count, format, slopeCount)
}

func QuickGelu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	defaultActivation.QuickGelu(dst, src, count, format)
}

func RReLU(dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	lower, upper float32) {
	defaultActivation.RReLU(dst, src, count, format, lower, upper)
}

func ReGLU(dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType) {
	defaultActivation.ReGLU(dst, packed, batch, halfCount, format)
}

func ReGLUTensors(dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType) {
	defaultActivation.ReGLUTensors(dst, gate, up, count, format)
}

func ReLU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	defaultActivation.ReLU(dst, src, count, format)
}

func SELU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	defaultActivation.SELU(dst, src, count, format)
}

func SeGLU(dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType) {
	defaultActivation.SeGLU(dst, packed, batch, halfCount, format)
}

func SeGLUTensors(dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType) {
	defaultActivation.SeGLUTensors(dst, gate, up, count, format)
}

func SiGLU(dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType) {
	defaultActivation.SiGLU(dst, packed, batch, halfCount, format)
}

func SiGLUTensors(dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType) {
	defaultActivation.SiGLUTensors(dst, gate, up, count, format)
}

func Sigmoid(dst, src unsafe.Pointer, count int, format dtype.DType) {
	defaultActivation.Sigmoid(dst, src, count, format)
}

func Silu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	defaultActivation.Silu(dst, src, count, format)
}

func Snake(dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	alpha float32) {
	defaultActivation.Snake(dst, src, count, format, alpha)
}

func SnakeParametric(dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	alpha, beta float32) {
	defaultActivation.SnakeParametric(dst, src, count, format, alpha, beta)
}

func SoftShrink(dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	lambda float32) {
	defaultActivation.SoftShrink(dst, src, count, format, lambda)
}

func Softmax(dst, src unsafe.Pointer, count int, format dtype.DType) {
	defaultActivation.Softmax(dst, src, count, format)
}

func Softplus(dst, src unsafe.Pointer, count int, format dtype.DType) {
	defaultActivation.Softplus(dst, src, count, format)
}

func Softsign(dst, src unsafe.Pointer, count int, format dtype.DType) {
	defaultActivation.Softsign(dst, src, count, format)
}

func SwiGLU(dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType) {
	defaultActivation.SwiGLU(dst, packed, batch, halfCount, format)
}

func SwiGLUTensors(dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType) {
	defaultActivation.SwiGLUTensors(dst, gate, up, count, format)
}

func Swish(dst, src unsafe.Pointer, count int, format dtype.DType) {
	defaultActivation.Swish(dst, src, count, format)
}

func Tanh(dst, src unsafe.Pointer, count int, format dtype.DType) {
	defaultActivation.Tanh(dst, src, count, format)
}

func TanhShrink(dst, src unsafe.Pointer, count int, format dtype.DType) {
	defaultActivation.TanhShrink(dst, src, count, format)
}

func Threshold(dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	threshold float32) {
	defaultActivation.Threshold(dst, src, count, format, threshold)
}
