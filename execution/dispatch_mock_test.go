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

func (noopDeviceBackend) RMSNorm(input, scale, output unsafe.Pointer, rows, lastDim int, format dtype.DType) {
	panic("noopDeviceBackend.RMSNorm invoked")
}

func (noopDeviceBackend) LayerNorm(input, scale, bias, output unsafe.Pointer, rows, lastDim int, format dtype.DType) {
	panic("noopDeviceBackend.LayerNorm invoked")
}

func (noopDeviceBackend) Matmul(out, left, right unsafe.Pointer, rows, inner, cols int, format dtype.DType) {
	panic("noopDeviceBackend.Matmul invoked")
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

func (noopDeviceBackend) SwiGLUTensors(dst, gate, up unsafe.Pointer, count int, format dtype.DType) {
	panic("noopDeviceBackend.SwiGLUTensors invoked")
}

func (noopDeviceBackend) RoPE(config device.RoPEConfig, input, output unsafe.Pointer, seqLen, numHeads, headDim int, format dtype.DType) {
	panic("noopDeviceBackend.RoPE invoked")
}

func (noopDeviceBackend) MultiHeadAttention(config device.MultiHeadAttentionConfig, query, key, value, output unsafe.Pointer, seqQ, seqK int, format dtype.DType) {
	panic("noopDeviceBackend.MultiHeadAttention invoked")
}

var _ executionDevice = noopDeviceBackend{}
