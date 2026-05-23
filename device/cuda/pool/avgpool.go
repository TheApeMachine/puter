//go:build cuda

package pool

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

func (pool *Pool) AvgPool2D(
	config device.PoolConfig,
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
) {
	pool.host.DispatchAvgPool2D(config, input, output, batch, channels, inHeight, inWidth, outHeight, outWidth, format)
}
