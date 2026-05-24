//go:build xla

package pool

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

func (pool *Pool) MaxPool2D(
	config device.PoolConfig,
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
) {
	pool.host.DispatchMaxPool2D(config, input, output, batch, channels, inHeight, inWidth, outHeight, outWidth, format)
}

func (pool *Pool) AvgPool2D(
	config device.PoolConfig,
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
) {
	pool.host.DispatchAvgPool2D(config, input, output, batch, channels, inHeight, inWidth, outHeight, outWidth, format)
}

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
