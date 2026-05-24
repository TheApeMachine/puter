package normalization

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

/*
Normalization implements device.Normalization for the XLA backend.
*/
type Normalization struct {
	host Host
}

/*
Host is the XLA dispatch surface normalization operations call into.
*/
type Host interface {
	NeedsPlatform()
	NotImplemented(methodName string)
	DispatchBatchNormEval(
		input, scale, bias, mean, variance, output unsafe.Pointer,
		batch, channels, spatial int,
		format dtype.DType,
	)
	DispatchGroupNorm(
		config device.GroupNormConfig,
		input, scale, bias, output unsafe.Pointer,
		batch, channels, spatial int,
		format dtype.DType,
	)
	DispatchInstanceNorm(
		input, scale, bias, output unsafe.Pointer,
		batch, channels, spatial int,
		format dtype.DType,
	)
}

/*
New wires a Normalization receiver to its XLA dispatch host.
*/
func New(host Host) Normalization {
	return Normalization{host: host}
}

func (normalization *Normalization) stubHost() {
	normalization.host.NeedsPlatform()
}

func (normalization *Normalization) unimplemented(methodName string) {
	normalization.host.NotImplemented(methodName)
}
