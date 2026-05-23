//go:build !xla

package pool

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)
func (pool *Pool) MaxPool2D( config PoolConfig, input, output unsafe.Pointer, batch, channels, inHeight, inWidth, outHeight, outWidth int, format dtype.DType, ) {
	pool.stubHost()
}

func (pool *Pool) AvgPool2D( config PoolConfig, input, output unsafe.Pointer, batch, channels, inHeight, inWidth, outHeight, outWidth int, format dtype.DType, ) {
	pool.stubHost()
}

func (pool *Pool) AdaptiveMaxPool2D( input, output unsafe.Pointer, batch, channels, inHeight, inWidth, outHeight, outWidth int, format dtype.DType, ) {
	pool.stubHost()
}

func (pool *Pool) AdaptiveAvgPool2D( input, output unsafe.Pointer, batch, channels, inHeight, inWidth, outHeight, outWidth int, format dtype.DType, ) {
	pool.stubHost()
}

