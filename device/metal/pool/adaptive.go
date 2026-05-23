//go:build darwin && cgo

package pool

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (pool *Pool) AdaptiveMaxPool2D(
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
) {
	pool.host.DispatchAdaptiveMaxPool2D(input, output, batch, channels, inHeight, inWidth, outHeight, outWidth, format)
}

func (pool *Pool) AdaptiveAvgPool2D(
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
) {
	pool.host.DispatchAdaptiveAvgPool2D(input, output, batch, channels, inHeight, inWidth, outHeight, outWidth, format)
}
