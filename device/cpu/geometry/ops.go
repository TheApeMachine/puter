package geometry

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

func (geometry Geometry) PhaseCoupling(
	destination, leftGrowth, rightGrowth unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	if count == 0 {
		return
	}

	switch format {
	case dtype.Float32:
		runPhaseCouplingFloat32(destination, leftGrowth, rightGrowth, count)
	case dtype.Float16:
		runPhaseCouplingFloat16(destination, leftGrowth, rightGrowth, count)
	case dtype.BFloat16:
		runPhaseCouplingBFloat16(destination, leftGrowth, rightGrowth, count)
	default:
		panic("geometry: unsupported dtype")
	}
}

func (geometry Geometry) PhaseVelocity(
	destination, current, previous unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	if count == 0 {
		return
	}

	switch format {
	case dtype.Float32:
		runPhaseVelocityFloat32(destination, current, previous, count)
	case dtype.Float16:
		runPhaseVelocityFloat16(destination, current, previous, count)
	case dtype.BFloat16:
		runPhaseVelocityBFloat16(destination, current, previous, count)
	default:
		panic("geometry: unsupported dtype")
	}
}

func (geometry Geometry) GeometricProduct(
	destination, left, right unsafe.Pointer,
) {
	geometricProductKernel(
		(*float64)(left),
		(*float64)(right),
		(*float64)(destination),
	)
}

func (geometry Geometry) PhaseDialNormalize(
	dial unsafe.Pointer,
) {
	dialNormalize128Native(phaseDialView(dial))
}

func (geometry Geometry) PhaseDialSimilarity(
	destination, left, right unsafe.Pointer,
) {
	similarity := dialSimilarity128Native(phaseDialView(left), phaseDialView(right))
	writeFloat64(destination, similarity)
}

func (geometry Geometry) PhaseDialRotate(
	destination, source, cosine, sine unsafe.Pointer,
) {
	dialRotate128FromTrigNative(
		phaseDialView(destination),
		phaseDialView(source),
		readFloat64(cosine),
		readFloat64(sine),
	)
}

func (geometry Geometry) PhaseDialAddPhases(
	dial, cosines, sines unsafe.Pointer,
) {
	dialAddPhases128Native(
		phaseDialView(dial),
		float64View(cosines, device.PhaseDialDimensions),
		float64View(sines, device.PhaseDialDimensions),
	)
}

func (geometry Geometry) PhaseDialComposeMidpoint(
	destination, left, right unsafe.Pointer,
) {
	result := dialComposeMidpoint128Native(phaseDialView(left), phaseDialView(right))
	copy(phaseDialView(destination), result)
}

func (geometry Geometry) PhaseRotorSimilarity(
	destination, left, right unsafe.Pointer,
) {
	similarity := rotorSimilarityAverage(phaseRotorView(left), phaseRotorView(right))
	writeFloat64(destination, similarity)
}

func (geometry Geometry) EigenToroidalFromTags(
	phaseDestination, frequencyDestination, tags unsafe.Pointer,
	tagCount, windowSize int,
) {
	if tagCount == 0 {
		return
	}

	eigenToroidalFromTags(
		float64View(phaseDestination, device.EigenSymbolDimensions),
		float64View(frequencyDestination, device.EigenSymbolDimensions),
		unsafe.Slice((*uint64)(tags), tagCount),
		windowSize,
	)
}

func (geometry Geometry) EigenCircularMeanPhase(
	destination, phaseTable, sequence unsafe.Pointer,
	sequenceLength int,
) {
	if sequenceLength == 0 {
		return
	}

	meanPhase := eigenCircularMeanPhase(
		float64View(phaseTable, device.EigenSymbolDimensions),
		unsafe.Slice((*byte)(sequence), sequenceLength),
	)
	writeFloat64(destination, meanPhase)
}

func phaseDialView(base unsafe.Pointer) PhaseDial {
	return unsafe.Slice((*complex128)(base), device.PhaseDialDimensions)
}

func phaseRotorView(base unsafe.Pointer) PhaseRotor {
	return unsafe.Slice((*Multivector)(base), device.PhaseDialDimensions)
}

func float64View(base unsafe.Pointer, count int) []float64 {
	return unsafe.Slice((*float64)(base), count)
}

func readFloat64(source unsafe.Pointer) float64 {
	return *(*float64)(source)
}

func writeFloat64(destination unsafe.Pointer, value float64) {
	*(*float64)(destination) = value
}
