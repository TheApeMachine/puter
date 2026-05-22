//go:build !darwin || !cgo

package metal

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device"
)

var _ device.Backend = (*Backend)(nil)

func (backend *Backend) deviceNeedsPlatform() {
	if backend.bridge == nil {
		panic(tensor.ErrNeedsPlatformSetup)
	}
}

func (backend *Backend) CountString(counts *[8]int, str string) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Count8(counts *[8]int, buf []uint8) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Count16(counts *[16]int, buf []uint16) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Count32(counts *[32]int, buf []uint32) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Count64(counts *[64]int, buf []uint64) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Exp(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Log(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Log1p(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Expm1(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Sigmoid(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) LogSigmoid(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Tanh(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Silu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Swish(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) GeluTanh(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Gelu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) ReLU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) LeakyReLU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) ELU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) CELU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) SELU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Softplus(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Mish(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Softsign(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) HardSigmoid(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) HardSwish(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) HardTanh(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) HardGelu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) QuickGelu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) TanhShrink(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Softmax(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) LogSoftmax(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) PReLU(dst, src unsafe.Pointer, count int, format dtype.DType, negativeSlope float32) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) PReLUV(dst, src, slopes unsafe.Pointer, count int, format dtype.DType, slopeCount int) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) LeakyReLUSlope(dst, src unsafe.Pointer, count int, format dtype.DType, negativeSlope float32) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) ELUAlpha(dst, src unsafe.Pointer, count int, format dtype.DType, alpha float32) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) CELUAlpha(dst, src unsafe.Pointer, count int, format dtype.DType, alpha float32) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Threshold(dst, src unsafe.Pointer, count int, format dtype.DType, threshold float32) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) HardTanhRange(dst, src unsafe.Pointer, count int, format dtype.DType, minVal, maxVal float32) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Snake(dst, src unsafe.Pointer, count int, format dtype.DType, alpha float32) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) SnakeParametric(dst, src unsafe.Pointer, count int, format dtype.DType, alpha, beta float32) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) HardShrink(dst, src unsafe.Pointer, count int, format dtype.DType, lambda float32) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) SoftShrink(dst, src unsafe.Pointer, count int, format dtype.DType, lambda float32) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) RReLU(dst, src unsafe.Pointer, count int, format dtype.DType, lower, upper float32) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) GLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) GeGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) GeGLUTanh(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) SwiGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) ReGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) SiGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) GLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) GeGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) GeGLUTanhTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) SwiGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) ReGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) SiGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) LinGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) SeGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) LinGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) SeGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Add(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Sub(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Mul(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Div(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Max(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Min(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Abs(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Neg(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Sqrt(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Axpy(y, x unsafe.Pointer, count int, alpha float32, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Sum(values unsafe.Pointer, count int, format dtype.DType) float32 {
	backend.deviceNeedsPlatform()
	return 0
}

func (backend *Backend) Prod(values unsafe.Pointer, count int, format dtype.DType) float32 {
	backend.deviceNeedsPlatform()
	return 0
}

func (backend *Backend) ReduceMin(values unsafe.Pointer, count int, format dtype.DType) float32 {
	backend.deviceNeedsPlatform()
	return 0
}

func (backend *Backend) ReduceMax(values unsafe.Pointer, count int, format dtype.DType) float32 {
	backend.deviceNeedsPlatform()
	return 0
}

func (backend *Backend) L1Norm(values unsafe.Pointer, count int, format dtype.DType) float32 {
	backend.deviceNeedsPlatform()
	return 0
}
