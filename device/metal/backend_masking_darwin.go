//go:build darwin && cgo

package metal

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func (backend *Backend) ApplyMask(
	input, mask, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	tensors := backend.tensorsAtPanic(input, mask, output)
	requireMaskingFormat(tensors[0].dtype, format)
	requireMaskingCount(count, tensors[0].shape.Len())
	devicePanic(runMetalApplyMask(tensors[0], tensors[1], tensors[2]))
}

func (backend *Backend) CausalMask(
	output unsafe.Pointer,
	seqQ, seqK int,
	format dtype.DType,
) {
	tensors := backend.tensorsAtPanic(output)
	requireMaskingFormat(tensors[0].dtype, format)
	requireMaskingMatrixDims(tensors[0].shape, seqQ, seqK)
	devicePanic(runMetalCausalMask(tensors[0], tensors[0]))
}

func (backend *Backend) ALiBiBias(
	scores, slope, output unsafe.Pointer,
	seqQ, seqK int,
	format dtype.DType,
) {
	tensors := backend.tensorsAtPanic(scores, slope, output)
	requireMaskingFormat(tensors[0].dtype, format)
	requireMaskingMatrixDims(tensors[0].shape, seqQ, seqK)
	devicePanic(runMetalALiBiBias(tensors[0], tensors[1], tensors[2]))
}

func requireMaskingFormat(storageDType dtype.DType, format dtype.DType) {
	if storageDType != format {
		devicePanic(tensor.ErrShapeMismatch)
	}
}

func requireMaskingCount(count int, tensorLen int) {
	if count != tensorLen {
		devicePanic(tensor.ErrShapeMismatch)
	}
}

func requireMaskingMatrixDims(shape tensor.Shape, seqQ int, seqK int) {
	dims := shape.Dims()

	if len(dims) != 2 || dims[0] != seqQ || dims[1] != seqK {
		devicePanic(tensor.ErrShapeMismatch)
	}
}
