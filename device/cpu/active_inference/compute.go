package active_inference

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func dispatchFreeEnergy(
	likelihood, posterior, prior, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	switch format {
	case dtype.Float32:
		runFreeEnergyF32(likelihood, posterior, prior, output, count)
	case dtype.BFloat16:
		runFreeEnergyBF16(likelihood, posterior, prior, output, count)
	case dtype.Float16:
		runFreeEnergyF16(likelihood, posterior, prior, output, count)
	default:
		panic("active_inference: unsupported dtype")
	}
}

func dispatchExpectedFreeEnergy(
	predictedObs, preferredObs, predictedState, output unsafe.Pointer,
	obsCount, stateCount int,
	format dtype.DType,
) {
	switch format {
	case dtype.Float32:
		runExpectedFreeEnergyF32(
			predictedObs, preferredObs, predictedState, output,
			obsCount, stateCount,
		)
	case dtype.BFloat16:
		runExpectedFreeEnergyBF16(
			predictedObs, preferredObs, predictedState, output,
			obsCount, stateCount,
		)
	case dtype.Float16:
		runExpectedFreeEnergyF16(
			predictedObs, preferredObs, predictedState, output,
			obsCount, stateCount,
		)
	default:
		panic("active_inference: unsupported dtype")
	}
}

func dispatchBeliefUpdate(
	likelihood, prior, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	switch format {
	case dtype.Float32:
		runBeliefUpdateF32(likelihood, prior, output, count)
	case dtype.BFloat16:
		runBeliefUpdateBF16(likelihood, prior, output, count)
	case dtype.Float16:
		runBeliefUpdateF16(likelihood, prior, output, count)
	default:
		panic("active_inference: unsupported dtype")
	}
}

func dispatchPrecisionWeight(
	errors, precision, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	switch format {
	case dtype.Float32:
		runPrecisionWeightF32(errors, precision, output, count)
	case dtype.BFloat16:
		runPrecisionWeightBF16(errors, precision, output, count)
	case dtype.Float16:
		runPrecisionWeightF16(errors, precision, output, count)
	default:
		panic("active_inference: unsupported dtype")
	}
}

func runFreeEnergyF32(
	likelihood, posterior, prior, output unsafe.Pointer,
	count int,
) {
	likelihoodView := unsafe.Slice((*float32)(likelihood), count)
	posteriorView := unsafe.Slice((*float32)(posterior), count)
	priorView := unsafe.Slice((*float32)(prior), count)
	outputView := unsafe.Slice((*float32)(output), 1)

	outputView[0] = FreeEnergyFloat32Native(likelihoodView, posteriorView, priorView)
}

func runExpectedFreeEnergyF32(
	predictedObs, preferredObs, predictedState, output unsafe.Pointer,
	obsCount, stateCount int,
) {
	predictedObsView := unsafe.Slice((*float32)(predictedObs), obsCount)
	preferredObsView := unsafe.Slice((*float32)(preferredObs), obsCount)
	predictedStateView := unsafe.Slice((*float32)(predictedState), stateCount)
	outputView := unsafe.Slice((*float32)(output), 1)

	outputView[0] = ExpectedFreeEnergyFloat32Native(
		predictedObsView, preferredObsView, predictedStateView,
	)
}

func runBeliefUpdateF32(
	likelihood, prior, output unsafe.Pointer,
	count int,
) {
	likelihoodView := unsafe.Slice((*float32)(likelihood), count)
	priorView := unsafe.Slice((*float32)(prior), count)
	outputView := unsafe.Slice((*float32)(output), count)

	BeliefUpdateFloat32Native(likelihoodView, priorView, outputView)
}

func runPrecisionWeightF32(
	errors, precision, output unsafe.Pointer,
	count int,
) {
	errorsView := unsafe.Slice((*float32)(errors), count)
	precisionView := unsafe.Slice((*float32)(precision), count)
	outputView := unsafe.Slice((*float32)(output), count)

	PrecisionWeightFloat32Native(errorsView, precisionView, outputView)
}

func runFreeEnergyBF16(
	likelihood, posterior, prior, output unsafe.Pointer,
	count int,
) {
	likelihoodView := unsafe.Slice((*dtype.BF16)(likelihood), count)
	posteriorView := unsafe.Slice((*dtype.BF16)(posterior), count)
	priorView := unsafe.Slice((*dtype.BF16)(prior), count)
	outputView := unsafe.Slice((*dtype.BF16)(output), 1)

	result := FreeEnergyBFloat16Native(likelihoodView, posteriorView, priorView)
	outputView[0] = result
}

func runExpectedFreeEnergyBF16(
	predictedObs, preferredObs, predictedState, output unsafe.Pointer,
	obsCount, stateCount int,
) {
	predictedObsView := unsafe.Slice((*dtype.BF16)(predictedObs), obsCount)
	preferredObsView := unsafe.Slice((*dtype.BF16)(preferredObs), obsCount)
	predictedStateView := unsafe.Slice((*dtype.BF16)(predictedState), stateCount)
	outputView := unsafe.Slice((*dtype.BF16)(output), 1)

	result := ExpectedFreeEnergyBFloat16Native(
		predictedObsView, preferredObsView, predictedStateView,
	)
	outputView[0] = result
}

func runBeliefUpdateBF16(
	likelihood, prior, output unsafe.Pointer,
	count int,
) {
	likelihoodView := unsafe.Slice((*dtype.BF16)(likelihood), count)
	priorView := unsafe.Slice((*dtype.BF16)(prior), count)
	outputView := unsafe.Slice((*dtype.BF16)(output), count)

	BeliefUpdateBFloat16Native(likelihoodView, priorView, outputView)
}

func runPrecisionWeightBF16(
	errors, precision, output unsafe.Pointer,
	count int,
) {
	errorsView := unsafe.Slice((*dtype.BF16)(errors), count)
	precisionView := unsafe.Slice((*dtype.BF16)(precision), count)
	outputView := unsafe.Slice((*dtype.BF16)(output), count)

	PrecisionWeightBFloat16Native(errorsView, precisionView, outputView)
}

func runFreeEnergyF16(
	likelihood, posterior, prior, output unsafe.Pointer,
	count int,
) {
	likelihoodView := unsafe.Slice((*dtype.F16)(likelihood), count)
	posteriorView := unsafe.Slice((*dtype.F16)(posterior), count)
	priorView := unsafe.Slice((*dtype.F16)(prior), count)
	outputView := unsafe.Slice((*dtype.F16)(output), 1)

	result := FreeEnergyFloat16Native(likelihoodView, posteriorView, priorView)
	outputView[0] = result
}

func runExpectedFreeEnergyF16(
	predictedObs, preferredObs, predictedState, output unsafe.Pointer,
	obsCount, stateCount int,
) {
	predictedObsView := unsafe.Slice((*dtype.F16)(predictedObs), obsCount)
	preferredObsView := unsafe.Slice((*dtype.F16)(preferredObs), obsCount)
	predictedStateView := unsafe.Slice((*dtype.F16)(predictedState), stateCount)
	outputView := unsafe.Slice((*dtype.F16)(output), 1)

	result := ExpectedFreeEnergyFloat16Native(
		predictedObsView, preferredObsView, predictedStateView,
	)
	outputView[0] = result
}

func runBeliefUpdateF16(
	likelihood, prior, output unsafe.Pointer,
	count int,
) {
	likelihoodView := unsafe.Slice((*dtype.F16)(likelihood), count)
	priorView := unsafe.Slice((*dtype.F16)(prior), count)
	outputView := unsafe.Slice((*dtype.F16)(output), count)

	BeliefUpdateFloat16Native(likelihoodView, priorView, outputView)
}

func runPrecisionWeightF16(
	errors, precision, output unsafe.Pointer,
	count int,
) {
	errorsView := unsafe.Slice((*dtype.F16)(errors), count)
	precisionView := unsafe.Slice((*dtype.F16)(precision), count)
	outputView := unsafe.Slice((*dtype.F16)(output), count)

	PrecisionWeightFloat16Native(errorsView, precisionView, outputView)
}
