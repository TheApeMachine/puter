package physics

import (
	"unsafe"
	"github.com/theapemachine/manifesto/dtype"
)

var defaultPhysics = New()

func BohmianVelocity(phase, output unsafe.Pointer,
	count int,
	spacing float32,
	format dtype.DType) {
	defaultPhysics.BohmianVelocity(phase, output, count, spacing, format)
}

func Divergence1D(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	defaultPhysics.Divergence1D(input, output, count, spacing, format)
}

func FFT1D(realIn, imagIn, realOut, imagOut unsafe.Pointer,
	count int,
	format dtype.DType) {
	defaultPhysics.FFT1D(realIn, imagIn, realOut, imagOut, count, format)
}

func Grad1D(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	defaultPhysics.Grad1D(input, output, count, spacing, format)
}

func IFFT1D(realIn, imagIn, realOut, imagOut unsafe.Pointer,
	count int,
	format dtype.DType) {
	defaultPhysics.IFFT1D(realIn, imagIn, realOut, imagOut, count, format)
}

func Laplacian(input, output unsafe.Pointer,
	dims []int,
	spacing float32,
	format dtype.DType) {
	defaultPhysics.Laplacian(input, output, dims, spacing, format)
}

func Laplacian4(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	defaultPhysics.Laplacian4(input, output, count, spacing, format)
}

func MadelungContinuity(density, velocity, residual unsafe.Pointer,
	count int,
	spacing float32,
	format dtype.DType) {
	defaultPhysics.MadelungContinuity(density, velocity, residual, count, spacing, format)
}

func QuantumPotential(density, output unsafe.Pointer,
	count int,
	spacing float32,
	format dtype.DType) {
	defaultPhysics.QuantumPotential(density, output, count, spacing, format)
}
