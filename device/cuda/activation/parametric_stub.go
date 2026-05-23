//go:build !cuda

package activation

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (activation *Activation) PReLU(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	negativeSlope float32,
) {
	activation.stubHost()
}

func (activation *Activation) PReLUV(
	dst, src, slopes unsafe.Pointer,
	count int,
	format dtype.DType,
	slopeCount int,
) {
	activation.stubHost()
}

func (activation *Activation) LeakyReLUSlope(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	negativeSlope float32,
) {
	activation.stubHost()
}

func (activation *Activation) ELUAlpha(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	alpha float32,
) {
	activation.stubHost()
}

func (activation *Activation) CELUAlpha(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	alpha float32,
) {
	activation.stubHost()
}

func (activation *Activation) Threshold(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	threshold float32,
) {
	activation.stubHost()
}

func (activation *Activation) HardTanhRange(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	minVal, maxVal float32,
) {
	activation.stubHost()
}

func (activation *Activation) Snake(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	alpha float32,
) {
	activation.stubHost()
}

func (activation *Activation) SnakeParametric(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	alpha, beta float32,
) {
	activation.stubHost()
}

func (activation *Activation) HardShrink(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	lambda float32,
) {
	activation.stubHost()
}

func (activation *Activation) SoftShrink(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	lambda float32,
) {
	activation.stubHost()
}

func (activation *Activation) RReLU(
	dst, src unsafe.Pointer,
	count int,
	format dtype.DType,
	lower, upper float32,
) {
	activation.stubHost()
}
