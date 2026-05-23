//go:build darwin && cgo

package activation

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

const (
	paramKernelPReLUSlope     = "prelu_slope"
	paramKernelLeakyReLUSlope = "leaky_relu_slope"
	paramKernelELUAlpha       = "elu_alpha"
	paramKernelCELUAlpha      = "celu_alpha"
	paramKernelThreshold      = "threshold"
)

func (activation *Activation) PReLU(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	negativeSlope float32,
) {
	_ = count
	activation.host.UnaryParam(dst, src, format, paramKernelPReLUSlope, negativeSlope)
}

func (activation *Activation) PReLUV(
	dst, src, slopes unsafe.Pointer,
	count int,
	format dtype.DType,
	slopeCount int,
) {
	_ = count
	_ = slopeCount
	activation.host.PReLUV(dst, src, slopes, format)
}

func (activation *Activation) LeakyReLUSlope(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	negativeSlope float32,
) {
	_ = count
	activation.host.UnaryParam(dst, src, format, paramKernelLeakyReLUSlope, negativeSlope)
}

func (activation *Activation) ELUAlpha(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	alpha float32,
) {
	_ = count
	activation.host.UnaryParam(dst, src, format, paramKernelELUAlpha, alpha)
}

func (activation *Activation) CELUAlpha(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	alpha float32,
) {
	_ = count
	activation.host.UnaryParam(dst, src, format, paramKernelCELUAlpha, alpha)
}

func (activation *Activation) Threshold(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	threshold float32,
) {
	_ = count
	activation.host.UnaryParam(dst, src, format, paramKernelThreshold, threshold)
}

func (activation *Activation) HardTanhRange(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	minVal, maxVal float32,
) {
	_ = count
	_ = minVal
	_ = maxVal
	activation.host.StandardUnary(dst, src, format, StandardHardTanh)
}

func (activation *Activation) Snake(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	alpha float32,
) {
	_ = count
	activation.host.UnaryParam(dst, src, format, paramKernelELUAlpha, alpha)
}

func (activation *Activation) SnakeParametric(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	alpha, beta float32,
) {
	_ = count
	_ = beta
	activation.host.UnaryParam(dst, src, format, paramKernelELUAlpha, alpha)
}

func (activation *Activation) HardShrink(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	lambda float32,
) {
	_ = count
	activation.host.UnaryParam(dst, src, format, paramKernelThreshold, lambda)
}

func (activation *Activation) SoftShrink(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	lambda float32,
) {
	_ = count
	activation.host.UnaryParam(dst, src, format, paramKernelThreshold, lambda)
}

func (activation *Activation) RReLU(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	lower, upper float32,
) {
	_ = count
	_ = upper
	activation.host.UnaryParam(dst, src, format, paramKernelPReLUSlope, lower)
}
