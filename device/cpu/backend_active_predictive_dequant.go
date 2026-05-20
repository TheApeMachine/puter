package cpu

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
	"github.com/theapemachine/puter/device/cpu/active_inference"
	"github.com/theapemachine/puter/device/cpu/dequant"
	"github.com/theapemachine/puter/device/cpu/predictive_coding"
	"github.com/theapemachine/puter/device/cpu/quant"
)

func (backend *Backend) FreeEnergy(
	likelihood, posterior, prior, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	active_inference.FreeEnergy(likelihood, posterior, prior, output, count, format)
}

func (backend *Backend) ExpectedFreeEnergy(
	predictedObs, preferredObs, predictedState, output unsafe.Pointer,
	obsCount, stateCount int,
	format dtype.DType,
) {
	active_inference.ExpectedFreeEnergy(predictedObs, preferredObs, predictedState, output, obsCount, stateCount, format)
}

func (backend *Backend) BeliefUpdate(likelihood, prior, output unsafe.Pointer, count int, format dtype.DType) {
	active_inference.BeliefUpdate(likelihood, prior, output, count, format)
}

func (backend *Backend) PrecisionWeight(errors, precision, output unsafe.Pointer, count int, format dtype.DType) {
	active_inference.PrecisionWeight(errors, precision, output, count, format)
}

func (backend *Backend) Prediction(
	weights, representation, output unsafe.Pointer,
	outDim, inDim int,
	format dtype.DType,
) {
	predictive_coding.Prediction(weights, representation, output, outDim, inDim, format)
}

func (backend *Backend) PredictionError(
	observed, predicted, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	predictive_coding.PredictionError(observed, predicted, output, count, format)
}

func (backend *Backend) UpdateRepresentation(
	config device.PredictiveCodingConfig,
	weights, representation, predictionError, output unsafe.Pointer,
	outDim, inDim int,
	format dtype.DType,
) {
	predictive_coding.UpdateRepresentation(
		predictiveCodingConfig(config), weights, representation, predictionError, output,
		outDim, inDim, format,
	)
}

func (backend *Backend) UpdateWeights(
	config device.PredictiveCodingConfig,
	weights, representation, predictionError, output unsafe.Pointer,
	outDim, inDim int,
	format dtype.DType,
) {
	predictive_coding.UpdateWeights(
		predictiveCodingConfig(config), weights, representation, predictionError, output,
		outDim, inDim, format,
	)
}

func (backend *Backend) Dequant(
	dst, src unsafe.Pointer,
	count int,
	config device.DequantInt8Config,
	dstFormat, srcFormat dtype.DType,
) {
	dequant.Dequant(dst, src, count, config, dstFormat, srcFormat)
}

func (backend *Backend) Dequant4(
	dst, src unsafe.Pointer,
	pairCount int,
	config device.DequantInt4Config,
	dstFormat, srcFormat dtype.DType,
) {
	dequant.Dequant4(dst, src, pairCount, config, dstFormat, srcFormat)
}

func (backend *Backend) Quant(
	dst, src unsafe.Pointer,
	count int,
	config device.DequantInt8Config,
	dstFormat, srcFormat dtype.DType,
) {
	quant.Quant(dst, src, count, config, dstFormat, srcFormat)
}
