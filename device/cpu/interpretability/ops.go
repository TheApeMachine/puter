package interpretability

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

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

	destinationView := unsafe.Slice((*float32)(destination), count)
	baseView := unsafe.Slice((*float32)(base), count)
	directionView := unsafe.Slice((*float32)(direction), count)

	activationSteerFloat32Kernel(destinationView, baseView, directionView, coefficient, count)
}
