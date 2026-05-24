package pool

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

/*
Pool implements device.Pool for the XLA backend.
*/
type Pool struct {
	host Host
}

/*
Host is the XLA dispatch surface pool operations call into.
*/
type Host interface {
	NeedsPlatform()
	NotImplemented(methodName string)
	DispatchMaxPool2D(
		config device.PoolConfig,
		input, output unsafe.Pointer,
		batch, channels, inHeight, inWidth, outHeight, outWidth int,
		format dtype.DType,
	)
	DispatchAvgPool2D(
		config device.PoolConfig,
		input, output unsafe.Pointer,
		batch, channels, inHeight, inWidth, outHeight, outWidth int,
		format dtype.DType,
	)
	DispatchAdaptiveMaxPool2D(
		input, output unsafe.Pointer,
		batch, channels, inHeight, inWidth, outHeight, outWidth int,
		format dtype.DType,
	)
	DispatchAdaptiveAvgPool2D(
		input, output unsafe.Pointer,
		batch, channels, inHeight, inWidth, outHeight, outWidth int,
		format dtype.DType,
	)
}

/*
New wires a Pool receiver to its XLA dispatch host.
*/
func New(host Host) Pool {
	return Pool{host: host}
}

func (pool *Pool) stubHost() {
	pool.host.NeedsPlatform()
}

func (pool *Pool) unimplemented(methodName string) {
	pool.host.NotImplemented(methodName)
}
