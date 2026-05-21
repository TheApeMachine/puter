//go:build darwin && cgo

package metal

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

var _ device.Backend = (*Backend)(nil)

func (backend *Backend) Add(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	backend.binaryElementwise(dst, left, right, format, metalBinaryFloat32Add)
}

func (backend *Backend) Sub(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	backend.binaryElementwise(dst, left, right, format, metalBinaryFloat32Sub)
}

func (backend *Backend) Mul(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	backend.binaryElementwise(dst, left, right, format, metalBinaryFloat32Mul)
}

func (backend *Backend) Div(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	backend.binaryElementwise(dst, left, right, format, metalBinaryFloat32Div)
}

func (backend *Backend) Max(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	backend.binaryElementwise(dst, left, right, format, metalBinaryFloat32Max)
}

func (backend *Backend) Min(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	backend.binaryElementwise(dst, left, right, format, metalBinaryFloat32Min)
}

func (backend *Backend) Abs(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.unaryElementwise(dst, src, format, metalUnaryFloat32Abs)
}

func (backend *Backend) Neg(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.unaryElementwise(dst, src, format, metalUnaryFloat32Neg)
}

func (backend *Backend) Sqrt(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.unaryElementwise(dst, src, format, metalUnaryFloat32Sqrt)
}

func (backend *Backend) Matmul(
	out, left, right unsafe.Pointer,
	rows, inner, cols int,
	format dtype.DType,
) {
	_ = rows
	_ = inner
	_ = cols
	_ = format
	tensors, err := backend.tensorsAt(left, right, out)

	if err != nil {
		return
	}

	_ = runMetalMatMul(tensors[0], tensors[1], tensors[2])
}

func (backend *Backend) ReLU(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.unaryElementwise(dst, src, format, metalUnaryFloat32Relu)
}

func (backend *Backend) Gelu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.unaryElementwise(dst, src, format, metalUnaryFloat32Gelu)
}

func (backend *Backend) Tanh(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.unaryElementwise(dst, src, format, metalUnaryFloat32Tanh)
}

func (backend *Backend) Sigmoid(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.unaryElementwise(dst, src, format, metalUnaryFloat32Sigmoid)
}

func (backend *Backend) Silu(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.unaryElementwise(dst, src, format, metalUnaryFloat32Silu)
}

func (backend *Backend) Swish(dst, src unsafe.Pointer, count int, format dtype.DType) {
	backend.unaryElementwise(dst, src, format, metalUnaryFloat32Swish)
}

func (backend *Backend) Softmax(dst, src unsafe.Pointer, count int, format dtype.DType) {
	_ = format
	tensors, err := backend.tensorsAt(src, dst)

	if err != nil {
		return
	}

	_ = runMetalSoftmax(tensors[0], tensors[1])
}

func (backend *Backend) LayerNorm(
	input, scale, bias, output unsafe.Pointer,
	rows, lastDim int,
	format dtype.DType,
) {
	_ = rows
	_ = lastDim
	_ = format
	tensors, err := backend.tensorsAt(input, scale, bias, output)

	if err != nil {
		return
	}

	_ = runMetalLayerNorm(tensors[0], tensors[1], tensors[2], tensors[3])
}

func (backend *Backend) RMSNorm(
	input, scale, output unsafe.Pointer,
	rows, lastDim int,
	format dtype.DType,
) {
	_ = rows
	_ = lastDim
	_ = format
	tensors, err := backend.tensorsAt(input, scale, output)

	if err != nil {
		return
	}

	_ = runMetalRMSNorm(tensors[0], tensors[1], tensors[2])
}

func (backend *Backend) Lookup(
	table, indices, output unsafe.Pointer,
	vocab, hidden, indexCount int,
	format dtype.DType,
) {
	_ = vocab
	_ = hidden
	_ = indexCount
	_ = format
	tensors, err := backend.tensorsAt(table, indices, output)

	if err != nil {
		return
	}

	_ = runMetalEmbeddingLookup(tensors[0], tensors[1], tensors[2])
}

func (backend *Backend) binaryElementwise(
	dst, left, right unsafe.Pointer,
	format dtype.DType,
	operation metalBinaryFloat32Operation,
) {
	_ = format
	tensors, err := backend.tensorsAt(left, right, dst)

	if err != nil {
		return
	}

	_ = runMetalBinaryElementwise(operation, tensors[0], tensors[1], tensors[2])
}

func (backend *Backend) unaryElementwise(
	dst, src unsafe.Pointer,
	format dtype.DType,
	operation metalUnaryFloat32Operation,
) {
	_ = format
	tensors, err := backend.tensorsAt(src, dst)

	if err != nil {
		return
	}

	_ = runMetalUnaryElementwise(operation, tensors[0], tensors[1])
}
