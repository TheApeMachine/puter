//go:build xla

package physics

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)
func (physics *Physics) Laplacian(input, output unsafe.Pointer, dims []int, spacing float32, format dtype.DType) {
	physics.unimplemented("Laplacian")
}

func (physics *Physics) Laplacian4(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	physics.unimplemented("Laplacian4")
}

func (physics *Physics) Grad1D(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	physics.unimplemented("Grad1D")
}

func (physics *Physics) Divergence1D(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	physics.unimplemented("Divergence1D")
}

func (physics *Physics) FFT1D(realIn, imagIn, realOut, imagOut unsafe.Pointer, count int, format dtype.DType) {
	physics.unimplemented("FFT1D")
}

func (physics *Physics) IFFT1D(realIn, imagIn, realOut, imagOut unsafe.Pointer, count int, format dtype.DType) {
	physics.unimplemented("IFFT1D")
}

func (physics *Physics) QuantumPotential(density, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	physics.unimplemented("QuantumPotential")
}

func (physics *Physics) BohmianVelocity(phase, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	physics.unimplemented("BohmianVelocity")
}

func (physics *Physics) MadelungContinuity( density, velocity, residual unsafe.Pointer, count int, spacing float32, format dtype.DType, ) {
	physics.unimplemented("MadelungContinuity")
}

