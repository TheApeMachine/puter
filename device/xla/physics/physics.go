package physics

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
Physics implements device.Physics for the XLA backend.
*/
type Physics struct {
	host Host
}

/*
Host is the XLA dispatch surface physics operations call into.
*/
type Host interface {
	NeedsPlatform()
	DispatchFFT1D(realIn, imagIn, realOut, imagOut unsafe.Pointer, count int, format dtype.DType)
	DispatchIFFT1D(realIn, imagIn, realOut, imagOut unsafe.Pointer, count int, format dtype.DType)
	DispatchLaplacian(input, output unsafe.Pointer, dims []int, spacing float32, format dtype.DType)
	DispatchLaplacian4(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType)
	DispatchGrad1D(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType)
	DispatchDivergence1D(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType)
	DispatchQuantumPotential(density, output unsafe.Pointer, count int, spacing float32, format dtype.DType)
	DispatchBohmianVelocity(phase, output unsafe.Pointer, count int, spacing float32, format dtype.DType)
	DispatchMadelungContinuity(
		density, velocity, residual unsafe.Pointer,
		count int,
		spacing float32,
		format dtype.DType,
	)
	NotImplemented(string)
}

/*
New wires a Physics receiver to its XLA dispatch host.
*/
func New(host Host) Physics {
	return Physics{host: host}
}

func (receiver *Physics) stubHost() {
	receiver.host.NeedsPlatform()
}

func (receiver *Physics) unimplemented(methodName string) {
	receiver.host.NotImplemented(methodName)
}
