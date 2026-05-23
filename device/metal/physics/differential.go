//go:build darwin && cgo

package physics

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (physics *Physics) Laplacian(input, output unsafe.Pointer, dims []int, spacing float32, format dtype.DType) {
	physics.host.DispatchLaplacian(input, output, dims, spacing, format)
}

func (physics *Physics) Laplacian4(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	physics.host.DispatchLaplacian4(input, output, count, spacing, format)
}

func (physics *Physics) Grad1D(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	physics.host.DispatchGrad1D(input, output, count, spacing, format)
}

func (physics *Physics) Divergence1D(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	physics.host.DispatchDivergence1D(input, output, count, spacing, format)
}

func (physics *Physics) QuantumPotential(density, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	physics.host.DispatchQuantumPotential(density, output, count, spacing, format)
}

func (physics *Physics) BohmianVelocity(phase, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	physics.host.DispatchBohmianVelocity(phase, output, count, spacing, format)
}

func (physics *Physics) MadelungContinuity(
	density, velocity, residual unsafe.Pointer,
	count int,
	spacing float32,
	format dtype.DType,
) {
	physics.host.DispatchMadelungContinuity(density, velocity, residual, count, spacing, format)
}
