//go:build !darwin || !cgo

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
	activation.stubHost()
}

func (activation *Activation) GeGLU(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	activation.stubHost()
}

func (activation *Activation) GeGLUTanh(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	activation.stubHost()
}

func (activation *Activation) SwiGLU(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	activation.stubHost()
}

func (activation *Activation) ReGLU(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	activation.stubHost()
}

func (activation *Activation) SiGLU(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	activation.stubHost()
}

func (activation *Activation) LinGLU(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	activation.stubHost()
}

func (activation *Activation) SeGLU(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	activation.stubHost()
}

func (activation *Activation) GLUTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	activation.stubHost()
}

func (activation *Activation) GeGLUTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	activation.stubHost()
}

func (activation *Activation) GeGLUTanhTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	activation.stubHost()
}

func (activation *Activation) SwiGLUTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	activation.stubHost()
}

func (activation *Activation) ReGLUTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	activation.stubHost()
}

func (activation *Activation) SiGLUTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	activation.stubHost()
}

func (activation *Activation) LinGLUTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	activation.stubHost()
}

func (activation *Activation) SeGLUTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	activation.stubHost()
}
