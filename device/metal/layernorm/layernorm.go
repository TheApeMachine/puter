package layernorm

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
Norm implements device.LayerNorm for the Metal backend.
*/
type Norm struct {
	host Host
}

/*
New wires a Norm receiver to its Metal dispatch host.
*/
func New(host Host) Norm {
	return Norm{host: host}
}

/*
Host is the Metal dispatch surface layer normalization operations call into.
*/
type Host interface {
	NeedsPlatform()
	LaunchLayerNorm(
		input, scale, bias, output unsafe.Pointer,
		rows, lastDim int,
		format dtype.DType,
	)
	LaunchRMSNorm(
		input, scale, output unsafe.Pointer,
		rows, lastDim int,
		format dtype.DType,
	)
}
