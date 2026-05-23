//go:build xla

package pool

import (
	"unsafe"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

func (pool *Pool) AdaptiveMaxPool2D(input, output unsafe.Pointer, batch, channels, inHeight, inWidth, outHeight, outWidth int, format dtype.DType,) {
	pool.unimplemented("AdaptiveMaxPool2D")
}

func (pool *Pool) AdaptiveAvgPool2D(input, output unsafe.Pointer, batch, channels, inHeight, inWidth, outHeight, outWidth int, format dtype.DType,) {
	pool.unimplemented("AdaptiveAvgPool2D")
}

func (pool *Pool) AvgPool2D(config device.PoolConfig, input, output unsafe.Pointer, batch, channels, inHeight, inWidth, outHeight, outWidth int, format dtype.DType,) {
	pool.unimplemented("AvgPool2D")
}

func (pool *Pool) MaxPool2D(config device.PoolConfig, input, output unsafe.Pointer, batch, channels, inHeight, inWidth, outHeight, outWidth int, format dtype.DType,) {
	pool.unimplemented("MaxPool2D")
}

