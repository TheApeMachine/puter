package activation

import (
	"unsafe"

	"github.com/theapemachine/puter/device/cpu/math"
	"github.com/theapemachine/manifesto/dtype"
)

func (activation Activation) GLU(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	dispatchGatedPacked(dst, packed, batch, halfCount, format, gluPackedKernel, math.FastGLU32)
}

func (activation Activation) GeGLU(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	dispatchGatedPacked(dst, packed, batch, halfCount, format, geGLUPackedKernel, math.FastGeGLU32)
}

func (activation Activation) GeGLUTanh(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	dispatchGatedPacked(dst, packed, batch, halfCount, format, geGLUTanhPackedKernel, math.FastGeGLUTanh32)
}

func (activation Activation) SwiGLU(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	dispatchGatedPacked(dst, packed, batch, halfCount, format, swiGLUPackedKernel, math.FastSwiGLU32)
}

func (activation Activation) ReGLU(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	dispatchGatedPacked(dst, packed, batch, halfCount, format, reGLUPackedKernel, math.FastReGLU32)
}

func (activation Activation) SiGLU(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	dispatchGatedPacked(dst, packed, batch, halfCount, format, siGLUPackedKernel, math.FastSiGLU32)
}

func (activation Activation) GLUTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	dispatchGatedTensors(dst, gate, up, count, format, gluTensorsKernel, math.FastGLU32)
}

func (activation Activation) GeGLUTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	dispatchGatedTensors(dst, gate, up, count, format, geGLUTensorsKernel, math.FastGeGLU32)
}

func (activation Activation) GeGLUTanhTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	dispatchGatedTensors(dst, gate, up, count, format, geGLUTanhTensorsKernel, math.FastGeGLUTanh32)
}

func (activation Activation) SwiGLUTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	dispatchGatedTensors(dst, gate, up, count, format, swiGLUTensorsKernel, math.FastSwiGLU32)
}

func (activation Activation) ReGLUTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	dispatchGatedTensors(dst, gate, up, count, format, reGLUTensorsKernel, math.FastReGLU32)
}

func (activation Activation) SiGLUTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	dispatchGatedTensors(dst, gate, up, count, format, siGLUTensorsKernel, math.FastSiGLU32)
}

func (activation Activation) LinGLU(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	dispatchGatedPacked(dst, packed, batch, halfCount, format, linGLUPackedKernel, math.FastLinGLU32)
}

func (activation Activation) SeGLU(
	dst, packed unsafe.Pointer,
	batch, halfCount int,
	format dtype.DType,
) {
	dispatchGatedPacked(dst, packed, batch, halfCount, format, seGLUPackedKernel, math.FastSeGLU32)
}

func (activation Activation) LinGLUTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	dispatchGatedTensors(dst, gate, up, count, format, linGLUTensorsKernel, math.FastLinGLU32)
}

func (activation Activation) SeGLUTensors(
	dst, gate, up unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	dispatchGatedTensors(dst, gate, up, count, format, seGLUTensorsKernel, math.FastSeGLU32)
}
