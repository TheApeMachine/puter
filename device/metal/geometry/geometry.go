package geometry

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	cpugeometry "github.com/theapemachine/puter/device/cpu/geometry"
)

/*
Geometry implements device.Geometry on Metal by delegating to the CPU
geometry kernels until dedicated MSL paths land.
*/
type Geometry struct {
	cpu cpugeometry.Geometry
}

func New() Geometry {
	return Geometry{cpu: cpugeometry.New()}
}

func (geometry Geometry) PhaseCoupling(
	destination, leftGrowth, rightGrowth unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	geometry.cpu.PhaseCoupling(destination, leftGrowth, rightGrowth, count, format)
}

func (geometry Geometry) PhaseVelocity(
	destination, current, previous unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	geometry.cpu.PhaseVelocity(destination, current, previous, count, format)
}

func (geometry Geometry) GeometricProduct(
	destination, left, right unsafe.Pointer,
) {
	geometry.cpu.GeometricProduct(destination, left, right)
}

func (geometry Geometry) PhaseDialNormalize(
	dial unsafe.Pointer,
) {
	geometry.cpu.PhaseDialNormalize(dial)
}

func (geometry Geometry) PhaseDialSimilarity(
	destination, left, right unsafe.Pointer,
) {
	geometry.cpu.PhaseDialSimilarity(destination, left, right)
}

func (geometry Geometry) PhaseDialRotate(
	destination, source, cosine, sine unsafe.Pointer,
) {
	geometry.cpu.PhaseDialRotate(destination, source, cosine, sine)
}

func (geometry Geometry) PhaseDialAddPhases(
	dial, cosines, sines unsafe.Pointer,
) {
	geometry.cpu.PhaseDialAddPhases(dial, cosines, sines)
}

func (geometry Geometry) PhaseDialComposeMidpoint(
	destination, left, right unsafe.Pointer,
) {
	geometry.cpu.PhaseDialComposeMidpoint(destination, left, right)
}

func (geometry Geometry) PhaseRotorSimilarity(
	destination, left, right unsafe.Pointer,
) {
	geometry.cpu.PhaseRotorSimilarity(destination, left, right)
}

func (geometry Geometry) EigenToroidalFromTags(
	phaseDestination, frequencyDestination, tags unsafe.Pointer,
	tagCount, windowSize int,
) {
	geometry.cpu.EigenToroidalFromTags(
		phaseDestination, frequencyDestination, tags, tagCount, windowSize,
	)
}

func (geometry Geometry) EigenCircularMeanPhase(
	destination, phaseTable, sequence unsafe.Pointer,
	sequenceLength int,
) {
	geometry.cpu.EigenCircularMeanPhase(destination, phaseTable, sequence, sequenceLength)
}
