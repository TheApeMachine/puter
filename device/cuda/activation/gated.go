//go:build cuda

package activation

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (activation *Activation) GLU(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	activation.host.GLUPacked(dst, packed, batch, halfCount, format, GLU)
}

func (activation *Activation) GeGLU(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	activation.host.GLUPacked(dst, packed, batch, halfCount, format, GeGLU)
}

func (activation *Activation) GeGLUTanh(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	activation.host.GLUPacked(dst, packed, batch, halfCount, format, GeGLUTanh)
}

func (activation *Activation) SwiGLU(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	activation.host.GLUPacked(dst, packed, batch, halfCount, format, SwiGLU)
}

func (activation *Activation) ReGLU(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	activation.host.GLUPacked(dst, packed, batch, halfCount, format, ReGLU)
}

func (activation *Activation) SiGLU(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	activation.host.GLUPacked(dst, packed, batch, halfCount, format, SiGLU)
}

func (activation *Activation) LinGLU(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	activation.host.GLUPacked(dst, packed, batch, halfCount, format, LinGLU)
}

func (activation *Activation) SeGLU(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	activation.host.GLUPacked(dst, packed, batch, halfCount, format, SeGLU)
}

func (activation *Activation) GLUTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	_ = count
	activation.host.GLUTensors(dst, gate, up, format, GLU)
}

func (activation *Activation) GeGLUTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	_ = count
	activation.host.GLUTensors(dst, gate, up, format, GeGLU)
}

func (activation *Activation) GeGLUTanhTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	_ = count
	activation.host.GLUTensors(dst, gate, up, format, GeGLUTanh)
}

func (activation *Activation) SwiGLUTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	_ = count
	activation.host.GLUTensors(dst, gate, up, format, SwiGLU)
}

func (activation *Activation) ReGLUTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	_ = count
	activation.host.GLUTensors(dst, gate, up, format, ReGLU)
}

func (activation *Activation) SiGLUTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	_ = count
	activation.host.GLUTensors(dst, gate, up, format, SiGLU)
}

func (activation *Activation) LinGLUTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	_ = count
	activation.host.GLUTensors(dst, gate, up, format, LinGLU)
}

func (activation *Activation) SeGLUTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	_ = count
	activation.host.GLUTensors(dst, gate, up, format, SeGLU)
}
