package physics

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

type Physics struct {
	host Host
}

func New(host Host) Physics {
	return Physics{host: host}
}

type Host interface {
	NeedsPlatform()
	DispatchBohmianVelocity(phase, output unsafe.Pointer, count int, spacing float32, format dtype.DType)
	DispatchDivergence1D(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType)
	DispatchGrad1D(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType)
	DispatchFFT1D(realIn, imagIn, realOut, imagOut unsafe.Pointer, count int, format dtype.DType)
	DispatchIFFT1D(realIn, imagIn, realOut, imagOut unsafe.Pointer, count int, format dtype.DType)
	DispatchLaplacian(input, output unsafe.Pointer, dims []int, spacing float32, format dtype.DType)
	DispatchLaplacian4(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType)
	DispatchMadelungContinuity(
		density, velocity, residual unsafe.Pointer,
		count int,
		spacing float32,
		format dtype.DType,
	)
	DispatchQuantumPotential(density, output unsafe.Pointer, count int, spacing float32, format dtype.DType)
}
