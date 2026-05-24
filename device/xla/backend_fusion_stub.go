//go:build !xla

package xla

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

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
