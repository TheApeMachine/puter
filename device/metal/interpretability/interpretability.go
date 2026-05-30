package interpretability

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
Interpretability implements device.Interpretability on Metal.
*/
type Interpretability struct {
	host Host
}

/*
New wires an Interpretability receiver to its Metal dispatch host.
*/
func New(host Host) Interpretability {
	return Interpretability{host: host}
}

/*
Host is the Metal dispatch surface interpretability operations call into.
*/
type Host interface {
	NeedsPlatform()
	DispatchActivationSteer(
		destination, base, direction unsafe.Pointer,
		coefficient float32,
		count int,
		format dtype.DType,
	)
}

func requireActivationSteerFloat32(format dtype.DType) {
	if format != dtype.Float32 {
		panic("interpretability: unsupported dtype")
	}
}

/*
ActivationSteer writes destination[i] = base[i] + coefficient * direction[i].
*/
func (interpretability Interpretability) ActivationSteer(
	destination, base, direction unsafe.Pointer,
	coefficient float32,
	count int,
	format dtype.DType,
) {
	requireActivationSteerFloat32(format)

	if count == 0 {
		return
	}

	interpretability.host.DispatchActivationSteer(
		destination, base, direction, coefficient, count, format,
	)
}
