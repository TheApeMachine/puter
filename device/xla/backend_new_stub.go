//go:build !xla

package xla

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
NewBackend constructs an XLA backend. Returns ErrNeedsPlatformSetup
when built without the xla tag.
*/
func NewBackend() (*Backend, error) {
	return nil, openXLABridgeUnavailable()
}

func openXLABridgeUnavailable() error {
	_, err := openXLABridge(nil)
	return err
}

func (backend *Backend) MatmulBiasGelu(
	out, left, right, bias unsafe.Pointer,
	rows, inner, cols int,
	format dtype.DType,
) {
}

func (backend *Backend) LayernormResidual(
	out, input, scale, bias, residual unsafe.Pointer,
	rows, lastDim int,
	format dtype.DType,
) {
}

func (backend *Backend) BuilderCacheMetrics() CacheMetrics {
	return CacheMetrics{}
}

func (backend *Backend) uploadSparseCSR(
	shape tensor.Shape,
	valueDType dtype.DType,
	values []byte,
	indices []tensor.SparseIndex,
) (tensor.SparseTensor, error) {
	return nil, tensor.ErrNeedsPlatformSetup
}
