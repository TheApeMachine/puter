//go:build xla

package xla

import (
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func (backend *Backend) uploadSparseCSR(
	shape tensor.Shape,
	valueDType dtype.DType,
	values []byte,
	indices []tensor.SparseIndex,
) (tensor.SparseTensor, error) {
	nnz := nnzFromCSRIndices(indices)

	expectedBytes, err := valueDType.BytesFor(nnz)

	if err != nil {
		return nil, err
	}

	if expectedBytes != len(values) {
		return nil, tensor.ErrShapeMismatch
	}

	valueShape, err := tensor.NewShape([]int{nnz})

	if err != nil {
		return nil, err
	}

	valueTensor, err := backend.Upload(valueShape, valueDType, values)

	if err != nil {
		return nil, err
	}

	valueDevice, ok := valueTensor.(*DeviceTensor)

	if !ok {
		_ = valueTensor.Close()
		return nil, tensor.ErrLayoutUnsupported
	}

	rowPtrSource := lookupCSRIndex(indices, "row_ptr")
	colIdxSource := lookupCSRIndex(indices, "col_idx")

	if rowPtrSource == nil || colIdxSource == nil {
		_ = valueDevice.Close()
		return nil, tensor.ErrShapeMismatch
	}

	rowPtrDevice, err := requireBackendDeviceTensor(backend, rowPtrSource)

	if err != nil {
		_ = valueDevice.Close()
		return nil, err
	}

	colIdxDevice, err := requireBackendDeviceTensor(backend, colIdxSource)

	if err != nil {
		_ = valueDevice.Close()
		return nil, err
	}

	return newDeviceSparseCSR(
		backend,
		shape,
		valueDType,
		valueDevice,
		rowPtrDevice,
		colIdxDevice,
		nnz,
	), nil
}
