//go:build !xla

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
	return nil, tensor.ErrNeedsPlatformSetup
}
