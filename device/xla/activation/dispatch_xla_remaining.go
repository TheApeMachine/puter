//go:build xla

package activation

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)
func (activation *Activation) Softmax(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.unimplemented("Softmax")
}

func (activation *Activation) LogSoftmax(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.unimplemented("LogSoftmax")
}

func (activation *Activation) PReLU(dst, src unsafe.Pointer, count int, format dtype.DType, negativeSlope float32) {
	activation.unimplemented("PReLU")
}

func (activation *Activation) PReLUV(dst, src, slopes unsafe.Pointer, count int, format dtype.DType, slopeCount int) {
	activation.unimplemented("PReLUV")
}

func (activation *Activation) ELUAlpha(dst, src unsafe.Pointer, count int, format dtype.DType, alpha float32) {
	activation.unimplemented("ELUAlpha")
}

func (activation *Activation) CELUAlpha(dst, src unsafe.Pointer, count int, format dtype.DType, alpha float32) {
	activation.unimplemented("CELUAlpha")
}

func (activation *Activation) Threshold(dst, src unsafe.Pointer, count int, format dtype.DType, threshold float32) {
	activation.unimplemented("Threshold")
}

func (activation *Activation) HardTanhRange(dst, src unsafe.Pointer, count int, format dtype.DType, minVal, maxVal float32) {
	activation.unimplemented("HardTanhRange")
}

func (activation *Activation) Snake(dst, src unsafe.Pointer, count int, format dtype.DType, alpha float32) {
	activation.unimplemented("Snake")
}

func (activation *Activation) SnakeParametric(dst, src unsafe.Pointer, count int, format dtype.DType, alpha, beta float32) {
	activation.unimplemented("SnakeParametric")
}

func (activation *Activation) HardShrink(dst, src unsafe.Pointer, count int, format dtype.DType, lambda float32) {
	activation.unimplemented("HardShrink")
}

func (activation *Activation) SoftShrink(dst, src unsafe.Pointer, count int, format dtype.DType, lambda float32) {
	activation.unimplemented("SoftShrink")
}

func (activation *Activation) RReLU(dst, src unsafe.Pointer, count int, format dtype.DType, lower, upper float32) {
	activation.unimplemented("RReLU")
}

func (activation *Activation) GLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	activation.unimplemented("GLU")
}

func (activation *Activation) GeGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	activation.unimplemented("GeGLU")
}

func (activation *Activation) GeGLUTanh(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	activation.unimplemented("GeGLUTanh")
}

func (activation *Activation) SwiGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	activation.unimplemented("SwiGLU")
}

func (activation *Activation) ReGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	activation.unimplemented("ReGLU")
}

func (activation *Activation) SiGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	activation.unimplemented("SiGLU")
}

func (activation *Activation) GLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	activation.unimplemented("GLUTensors")
}

func (activation *Activation) GeGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	activation.unimplemented("GeGLUTensors")
}

func (activation *Activation) GeGLUTanhTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	activation.unimplemented("GeGLUTanhTensors")
}

func (activation *Activation) SwiGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	activation.unimplemented("SwiGLUTensors")
}

func (activation *Activation) ReGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	activation.unimplemented("ReGLUTensors")
}

func (activation *Activation) SiGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	activation.unimplemented("SiGLUTensors")
}

func (activation *Activation) LinGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	activation.unimplemented("LinGLU")
}

func (activation *Activation) SeGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	activation.unimplemented("SeGLU")
}

func (activation *Activation) LinGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	activation.unimplemented("LinGLUTensors")
}

func (activation *Activation) SeGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	activation.unimplemented("SeGLUTensors")
}

