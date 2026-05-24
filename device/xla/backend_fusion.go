//go:build xla

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
	host := &ComputeHost{bridge: backend.bridge, builder: backend.builder}
	host.MatmulBiasGeluLaunch(out, left, right, bias, rows, inner, cols, format)
}

func (backend *Backend) LayernormResidual(
	out, input, scale, bias, residual unsafe.Pointer,
	rows, lastDim int,
	format dtype.DType,
) {
	host := &ComputeHost{bridge: backend.bridge, builder: backend.builder}
	host.LayernormResidualLaunch(out, input, scale, bias, residual, rows, lastDim, format)
}

func (backend *Backend) BuilderCacheMetrics() CacheMetrics {
	return backend.builder.CacheMetrics()
}
