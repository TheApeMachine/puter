package pool

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

/*
Pool implements device.Pool for the Metal backend.
Methods delegate kernel launch to a Host provided by the root Backend.
*/
type Pool struct {
	host Host
}

/*
New wires a Pool receiver to its Metal dispatch host.
*/
func New(host Host) Pool {
	return Pool{host: host}
}

/*
Host is the Metal dispatch surface pool operations call into.
*/
type Host interface {
	NeedsPlatform()
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
