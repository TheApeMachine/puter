package execution

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

/*
noopDeviceBackend is a minimal executionDevice used by tests that don't
exercise the device-call path. Every method panics — the test ensures
the dispatcher never reaches a device call (either by routing to a
codegen kernel or by failing earlier on a validation check).
*/
type noopDeviceBackend struct{}

func (noopDeviceBackend) Lookup(table, indices, output unsafe.Pointer, vocab, hidden, indexCount int, format dtype.DType) {
	panic("noopDeviceBackend.Lookup invoked")
}

func (noopDeviceBackend) TimestepEmbedding(config device.TimestepEmbeddingConfig, timesteps, output unsafe.Pointer, count, dim int, format dtype.DType) {
	panic("noopDeviceBackend.TimestepEmbedding invoked")
}

func (noopDeviceBackend) RMSNorm(config device.RMSNormConfig, input, scale, output unsafe.Pointer, rows, lastDim int, format dtype.DType) {
	panic("noopDeviceBackend.RMSNorm invoked")
}

func (noopDeviceBackend) AdaptiveRMSNorm(
	config device.RMSNormConfig,
	input, modulation, output unsafe.Pointer,
	rows, lastDim, rowsPerBatch, modulationCols int,
	format dtype.DType,
) {
	panic("noopDeviceBackend.AdaptiveRMSNorm invoked")
}

func (noopDeviceBackend) LayerNorm(input, scale, bias, output unsafe.Pointer, rows, lastDim int, format dtype.DType) {
	panic("noopDeviceBackend.LayerNorm invoked")
}

func (noopDeviceBackend) GroupNorm(
	config device.GroupNormConfig,
	input, scale, bias, output unsafe.Pointer,
	batch, channels, spatial int,
	format dtype.DType,
) {
	panic("noopDeviceBackend.GroupNorm invoked")
}

func (noopDeviceBackend) ModulatedLayerNorm(
	config device.ModulatedLayerNormConfig,
	input, modulation, output unsafe.Pointer,
	rows, lastDim, rowsPerBatch, modulationCols int,
	format dtype.DType,
) {
	panic("noopDeviceBackend.ModulatedLayerNorm invoked")
}

func (noopDeviceBackend) Matmul(out, left, right unsafe.Pointer, rows, inner, cols int, format dtype.DType) {
	panic("noopDeviceBackend.Matmul invoked")
}

func (noopDeviceBackend) Conv2D(
	config device.Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
	format dtype.DType,
) {
	panic("noopDeviceBackend.Conv2D invoked")
}

func (noopDeviceBackend) Add(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	panic("noopDeviceBackend.Add invoked")
}

func (noopDeviceBackend) Sub(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	panic("noopDeviceBackend.Sub invoked")
}

func (noopDeviceBackend) Mul(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	panic("noopDeviceBackend.Mul invoked")
}

func (noopDeviceBackend) Div(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	panic("noopDeviceBackend.Div invoked")
}

func (noopDeviceBackend) ReLU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	panic("noopDeviceBackend.ReLU invoked")
}

func (noopDeviceBackend) Sigmoid(dst, src unsafe.Pointer, count int, format dtype.DType) {
	panic("noopDeviceBackend.Sigmoid invoked")
}

func (noopDeviceBackend) Tanh(dst, src unsafe.Pointer, count int, format dtype.DType) {
	panic("noopDeviceBackend.Tanh invoked")
}

func (noopDeviceBackend) Gelu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	panic("noopDeviceBackend.Gelu invoked")
}

func (noopDeviceBackend) Silu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	panic("noopDeviceBackend.Silu invoked")
}

func (noopDeviceBackend) SwiGLU(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType) {
	panic("noopDeviceBackend.SwiGLU invoked")
}

func (noopDeviceBackend) SwiGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	panic("noopDeviceBackend.SwiGLUTensors invoked")
}

func (noopDeviceBackend) RoPE(config device.RoPEConfig, input, output unsafe.Pointer, seqLen, numHeads, headDim int, format dtype.DType) {
	panic("noopDeviceBackend.RoPE invoked")
}

func (noopDeviceBackend) MultiAxisRoPE(
	config device.MultiAxisRoPEConfig,
	input, output unsafe.Pointer,
	batch, seqLen, numHeads, headDim int,
	format dtype.DType,
) {
	panic("noopDeviceBackend.MultiAxisRoPE invoked")
}

func (noopDeviceBackend) MultiHeadAttention(config device.MultiHeadAttentionConfig, query, key, value, output unsafe.Pointer, seqQ, seqK int, format dtype.DType) {
	panic("noopDeviceBackend.MultiHeadAttention invoked")
}

func (noopDeviceBackend) ResonantUpdateForward(
	x, y, vr, vi, diag unsafe.Pointer,
	xOut, yOut, aOut, bOut, invROut unsafe.Pointer,
	batchTime, headCount, headDim int,
	config device.ResonantUpdateConfig,
	format dtype.DType,
) {
	panic("noopDeviceBackend.ResonantUpdateForward invoked")
}

func (noopDeviceBackend) ResonantUpdateBackward(
	gradXOut, gradYOut unsafe.Pointer,
	x, y, diag, a, b, invR unsafe.Pointer,
	gradX, gradY, gradVR, gradVI unsafe.Pointer,
	batchTime, headCount, headDim int,
	config device.ResonantUpdateConfig,
	format dtype.DType,
) {
	panic("noopDeviceBackend.ResonantUpdateBackward invoked")
}

var _ executionDevice = noopDeviceBackend{}

type batchRecordingDevice struct {
	noopDeviceBackend
	events   []string
	beginErr error
	endErr   error
}

func (deviceBackend *batchRecordingDevice) BeginBatch() error {
	deviceBackend.events = append(deviceBackend.events, "begin")
	return deviceBackend.beginErr
}

func (deviceBackend *batchRecordingDevice) EndBatch() error {
	deviceBackend.events = append(deviceBackend.events, "end")
	return deviceBackend.endErr
}

var _ executionDevice = (*batchRecordingDevice)(nil)
var _ batchExecutionDevice = (*batchRecordingDevice)(nil)
