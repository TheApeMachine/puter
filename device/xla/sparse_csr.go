//go:build xla

package xla

import (
	"context"
	"sync/atomic"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
DeviceSparseCSR is an XLA-resident CSR sparse tensor.
*/
type DeviceSparseCSR struct {
	backend  *Backend
	shape    tensor.Shape
	dtype    dtype.DType
	values   *DeviceTensor
	rowPtr   *DeviceTensor
	colIdx   *DeviceTensor
	nnz      int
	state    atomic.Uint32
	closed   atomic.Bool
	gradFlag atomic.Bool
}

func newDeviceSparseCSR(
	backend *Backend,
	shape tensor.Shape,
	valueDType dtype.DType,
	values *DeviceTensor,
	rowPtr *DeviceTensor,
	colIdx *DeviceTensor,
	nnz int,
) *DeviceSparseCSR {
	sparse := &DeviceSparseCSR{
		backend: backend,
		shape:   shape,
		dtype:   valueDType,
		values:  values,
		rowPtr:  rowPtr,
		colIdx:  colIdx,
		nnz:     nnz,
	}

	sparse.state.Store(uint32(tensor.StateReady))
	return sparse
}

func (sparse *DeviceSparseCSR) Shape() tensor.Shape { return sparse.shape }

func (sparse *DeviceSparseCSR) DType() dtype.DType { return sparse.dtype }

func (sparse *DeviceSparseCSR) Layout() tensor.Layout { return tensor.LayoutSparseCSR }

func (sparse *DeviceSparseCSR) Location() tensor.Location { return tensor.XLA }

func (sparse *DeviceSparseCSR) Len() int { return sparse.shape.Len() }

func (sparse *DeviceSparseCSR) Bytes() int {
	return sparse.values.Bytes() + sparse.rowPtr.Bytes() + sparse.colIdx.Bytes()
}

func (sparse *DeviceSparseCSR) State() tensor.State {
	return tensor.State(sparse.state.Load())
}

func (sparse *DeviceSparseCSR) WaitReady() error {
	if sparse.closed.Load() {
		return tensor.ErrTensorClosed
	}

	if err := sparse.values.WaitReady(); err != nil {
		return err
	}

	if err := sparse.rowPtr.WaitReady(); err != nil {
		return err
	}

	return sparse.colIdx.WaitReady()
}

func (sparse *DeviceSparseCSR) Sync(ctx context.Context) error {
	if err := sparse.WaitReady(); err != nil {
		return err
	}

	return ctx.Err()
}

func (sparse *DeviceSparseCSR) Ready() <-chan struct{} {
	ready := make(chan struct{})
	close(ready)
	return ready
}

func (sparse *DeviceSparseCSR) RequiresGrad() bool { return sparse.gradFlag.Load() }

func (sparse *DeviceSparseCSR) SetRequiresGrad(yes bool) error {
	if yes {
		return tensor.ErrNoAutograd
	}

	sparse.gradFlag.Store(false)
	return nil
}

func (sparse *DeviceSparseCSR) Grad() (tensor.Tensor, error) {
	return nil, tensor.ErrNoAutograd
}

func (sparse *DeviceSparseCSR) GradFn() tensor.GradFn { return nil }

func (sparse *DeviceSparseCSR) Close() error {
	if !sparse.closed.CompareAndSwap(false, true) {
		return nil
	}

	var closeErr error

	if sparse.values != nil {
		closeErr = sparse.values.Close()
	}

	if sparse.rowPtr != nil {
		if err := sparse.rowPtr.Close(); err != nil && closeErr == nil {
			closeErr = err
		}
	}

	if sparse.colIdx != nil {
		if err := sparse.colIdx.Close(); err != nil && closeErr == nil {
			closeErr = err
		}
	}

	sparse.state.Store(uint32(tensor.StateClosed))
	return closeErr
}

func (sparse *DeviceSparseCSR) RawBytes() (dtype.DType, []byte, error) {
	return dtype.Invalid, nil, tensor.ErrLayoutUnsupported
}

func (sparse *DeviceSparseCSR) Slice(start, length int) (tensor.Tensor, error) {
	return nil, tensor.ErrLayoutUnsupported
}

func (sparse *DeviceSparseCSR) Reshape(dims []int) (tensor.Tensor, error) {
	return nil, tensor.ErrLayoutUnsupported
}

func (sparse *DeviceSparseCSR) Float64Native() ([]float64, error) {
	return nil, tensor.ErrDTypeMismatch
}

func (sparse *DeviceSparseCSR) Float32Native() ([]float32, error) {
	return nil, tensor.ErrDTypeMismatch
}

func (sparse *DeviceSparseCSR) Float16Native() ([]dtype.F16, error) {
	return nil, tensor.ErrDTypeMismatch
}

func (sparse *DeviceSparseCSR) BFloat16Native() ([]dtype.BF16, error) {
	return nil, tensor.ErrDTypeMismatch
}

func (sparse *DeviceSparseCSR) Float8E4M3Native() ([]dtype.F8E4M3, error) {
	return nil, tensor.ErrDTypeMismatch
}

func (sparse *DeviceSparseCSR) Float8E5M2Native() ([]dtype.F8E5M2, error) {
	return nil, tensor.ErrDTypeMismatch
}

func (sparse *DeviceSparseCSR) Int64Native() ([]int64, error) {
	return nil, tensor.ErrDTypeMismatch
}

func (sparse *DeviceSparseCSR) Int32Native() ([]int32, error) {
	return nil, tensor.ErrDTypeMismatch
}

func (sparse *DeviceSparseCSR) Int16Native() ([]int16, error) {
	return nil, tensor.ErrDTypeMismatch
}

func (sparse *DeviceSparseCSR) Int8Native() ([]int8, error) {
	return nil, tensor.ErrDTypeMismatch
}

func (sparse *DeviceSparseCSR) Uint64Native() ([]uint64, error) {
	return nil, tensor.ErrDTypeMismatch
}

func (sparse *DeviceSparseCSR) Uint32Native() ([]uint32, error) {
	return nil, tensor.ErrDTypeMismatch
}

func (sparse *DeviceSparseCSR) Uint16Native() ([]uint16, error) {
	return nil, tensor.ErrDTypeMismatch
}

func (sparse *DeviceSparseCSR) Uint8Native() ([]uint8, error) {
	return nil, tensor.ErrDTypeMismatch
}

func (sparse *DeviceSparseCSR) BoolNative() (tensor.BitVector, error) {
	return tensor.BitVector{}, tensor.ErrDTypeMismatch
}

func (sparse *DeviceSparseCSR) Int4Native() (tensor.Int4Vector, error) {
	return tensor.Int4Vector{}, tensor.ErrDTypeMismatch
}

func (sparse *DeviceSparseCSR) NNZ() int { return sparse.nnz }

func (sparse *DeviceSparseCSR) Values() (tensor.Tensor, error) {
	if sparse.closed.Load() {
		return nil, tensor.ErrTensorClosed
	}

	return sparse.values, nil
}

func (sparse *DeviceSparseCSR) Indices() ([]tensor.SparseIndex, error) {
	if sparse.closed.Load() {
		return nil, tensor.ErrTensorClosed
	}

	return []tensor.SparseIndex{
		{Name: "row_ptr", Data: sparse.rowPtr},
		{Name: "col_idx", Data: sparse.colIdx},
	}, nil
}

func (sparse *DeviceSparseCSR) BlockSize() (rows, cols int, ok bool) {
	return 0, 0, false
}

func requireBackendDeviceTensor(backend *Backend, value tensor.Tensor) (*DeviceTensor, error) {
	deviceTensor, ok := value.(*DeviceTensor)

	if !ok {
		return nil, tensor.ErrLayoutUnsupported
	}

	if deviceTensor.backend != backend {
		return nil, tensor.ErrLayoutUnsupported
	}

	if err := deviceTensor.WaitReady(); err != nil {
		return nil, err
	}

	return deviceTensor, nil
}

func nnzFromCSRIndices(indices []tensor.SparseIndex) int {
	for _, candidate := range indices {
		if candidate.Name == "col_idx" && candidate.Data != nil {
			return candidate.Data.Len()
		}
	}

	return 0
}

func lookupCSRIndex(indices []tensor.SparseIndex, name string) tensor.Tensor {
	for _, candidate := range indices {
		if candidate.Name == name {
			return candidate.Data
		}
	}

	return nil
}

var _ tensor.SparseTensor = (*DeviceSparseCSR)(nil)
