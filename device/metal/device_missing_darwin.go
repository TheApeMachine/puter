//go:build darwin && cgo

package metal

import (
	"fmt"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device"
)

// #include "bridge_darwin.h"
import "C"

func (backend *Backend) ALiBiBias(scores, slope, output unsafe.Pointer, seqQ, seqK int, format dtype.DType) {
	tensors := backend.tensorsAtPanic(scores, slope, output)
	devicePanic(runMetalALiBiBias(tensors[0], tensors[1], tensors[2]))
}

func (backend *Backend) AdaptiveAvgPool2D(input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(input, output)
	devicePanic(runMetalAdaptiveAvgPool2D(tensors[0], tensors[1]))
}

func (backend *Backend) AdaptiveMaxPool2D(input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(input, output)
	devicePanic(runMetalAdaptiveMaxPool2D(tensors[0], tensors[1]))
}

func (backend *Backend) ApplyMask(input, mask, output unsafe.Pointer, count int, format dtype.DType) {
	tensors := backend.tensorsAtPanic(input, mask, output)
	devicePanic(runMetalApplyMask(tensors[0], tensors[1], tensors[2]))
}

func (backend *Backend) AvgPool2D(config device.PoolConfig,
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(input, output)
	devicePanic(runMetalAvgPool2D(tensors[0], tensors[1]))
}

func (backend *Backend) BackdoorAdjustment(conditional, marginalZ, output unsafe.Pointer,
	xCount, zCount, yCount int,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(conditional, marginalZ, output)
	devicePanic(runMetalBackdoorAdjustment(tensors[0], tensors[1], tensors[2]))
}

func (backend *Backend) Bag(table, indices, offsets, output unsafe.Pointer,
	vocab, hidden, bagCount, indexCount int,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(table, indices, offsets, output)
	devicePanic(runMetalEmbeddingBag(tensors[0], tensors[1], tensors[2], tensors[3]))
}

func (backend *Backend) BatchNormEval(input, scale, bias, mean, variance, output unsafe.Pointer,
	batch, channels, spatial int,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(input, scale, bias, mean, variance, output)
	devicePanic(runMetalBatchNormEval(tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5]))
}

func (backend *Backend) BeliefUpdate(likelihood, prior, output unsafe.Pointer, count int, format dtype.DType) {
	tensors := backend.tensorsAtPanic(likelihood, prior, output)
	devicePanic(runMetalBeliefUpdate(tensors[0], tensors[1], tensors[2]))
}

func (backend *Backend) Bind(left, right, output unsafe.Pointer, count int, format dtype.DType) {
	tensors := backend.tensorsAtPanic(left, right, output)
	devicePanic(runMetalVSABindKernel(tensors[0], tensors[1], tensors[2]))
}

func (backend *Backend) BohmianVelocity(phase, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	tensors := backend.tensorsAtPanic(phase, output)
	spacingTensor := backend.uploadFloat32Scalar(spacing, format)
	devicePanic(runMetalPhysicsBinary(metalPhysicsBohmianVelocity, tensors[0], spacingTensor, tensors[1]))
}

func (backend *Backend) Bundle(left, right, output unsafe.Pointer, count int, format dtype.DType) {
	tensors := backend.tensorsAtPanic(left, right, output)
	devicePanic(runMetalVSABundleKernel(tensors[0], tensors[1], tensors[2]))
}

func (backend *Backend) CATE(treated, control, output unsafe.Pointer, count int, format dtype.DType) {
	tensors := backend.tensorsAtPanic(treated, control, output)
	devicePanic(runMetalCATE(tensors[0], tensors[1], tensors[2]))
}

func (backend *Backend) CausalMask(output unsafe.Pointer, seqQ, seqK int, format dtype.DType) {
	tensors := backend.tensorsAtPanic(output)
	devicePanic(runMetalCausalMask(tensors[0], tensors[0]))
}

func runMetalCholesky(input tensor.Tensor, out tensor.Tensor) error {
	if out.Shape().Len() == 0 {
		return nil
	}

	metalInput, ok := input.(*metalTensor)
	if !ok {
		return fmt.Errorf("cholesky input must be a metalTensor")
	}

	metalOut, ok := out.(*metalTensor)
	if !ok {
		return fmt.Errorf("cholesky output must be a metalTensor")
	}

	token, err := metalCompletions.Begin(metalOut, metalInput)
	if err != nil {
		return err
	}

	status := C.MetalStatus{}
	dims := input.Shape().Dims()
	matrixOrder := dims[len(dims)-1]

	rc := C.metal_dispatch_cholesky(
		metalInput.bridge.device,
		metalInput.buffer,
		metalOut.buffer,
		C.uint32_t(matrixOrder),
		C.uint64_t(token),
		&status,
	)

	if rc != 0 {
		err := fmt.Errorf("metal cholesky: %s", metalStatus("dispatch", status))
		metalCompletions.Fail(token, err)
		return err
	}

	return nil
}

func (backend *Backend) Cholesky(input, output unsafe.Pointer, matrixOrder int, format dtype.DType) {
	tensors := backend.tensorsAtPanic(input, output)
	devicePanic(runMetalCholesky(tensors[0], tensors[1]))
}

func (backend *Backend) Conv1D(config device.Conv1DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inLength, outChannels, kernelLength, outLength int,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(input, weight, bias, output)
	devicePanic(runMetalConv1D(tensors[0], tensors[1], tensors[2], tensors[3]))
}

func (backend *Backend) Conv2D(config device.Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(input, weight, bias, output)
	devicePanic(runMetalConv2D(tensors[0], tensors[1], tensors[2], tensors[3]))
}

func (backend *Backend) Conv3D(config device.Conv3DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inD, inH, inW,
	outChannels, kD, kH, kW, outD, outH, outW int,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(input, weight, bias, output)
	devicePanic(runMetalConv3D(tensors[0], tensors[1], tensors[2], tensors[3]))
}

func (backend *Backend) ConvTranspose2D(config device.Conv2DConfig,
	input, weight, bias, output unsafe.Pointer,
	batch, inChannels, inHeight, inWidth,
	outChannels, kernelHeight, kernelWidth,
	outHeight, outWidth int,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(input, weight, bias, output)
	devicePanic(runMetalConvTranspose2D(tensors[0], tensors[1], tensors[2], tensors[3]))
}

func (backend *Backend) Counterfactual(observedY, observedX, counterfactualX, output unsafe.Pointer,
	count int,
	slope float32,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(observedY, observedX, counterfactualX, output)
	slopeTensor := backend.uploadFloat32Scalar(slope, format)
	devicePanic(runMetalCounterfactual(tensors[0], tensors[1], tensors[2], slopeTensor, tensors[3]))
}

func (backend *Backend) CrossEntropy(logits unsafe.Pointer,
	targets unsafe.Pointer,
	batchSize, classes int,
	format dtype.DType) float32 {
	tensors := backend.tensorsAtPanic(logits, targets)
	out := backend.emptyScalar(format)
	devicePanic(runMetalCrossEntropyLoss(tensors[0], tensors[1], out))
	return backend.readFloat32Scalar(out.residentPointer())
}

func (backend *Backend) DAGMarkovFactorization(conditionals unsafe.Pointer,
	conditionalCount int,
	output unsafe.Pointer,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(conditionals, output)

	// Create a dummy int32 tensor for parents to satisfy validation
	parents := backend.uploadInt32Scalar(0, dtype.Int32)

	devicePanic(runMetalDAGMarkovFactorization(tensors[0], parents, tensors[1]))
}

func (backend *Backend) Dequant(dst, src unsafe.Pointer, count int, config device.DequantInt8Config, dstFormat, srcFormat dtype.DType) {
	tensors := backend.tensorsAtPanic(src, dst)
	devicePanic(runMetalInt8Dequant(tensors[0], tensors[1]))
}

func (backend *Backend) Dequant4(dst, src unsafe.Pointer, pairCount int, config device.DequantInt4Config, dstFormat, srcFormat dtype.DType) {
	tensors := backend.tensorsAtPanic(src, dst)
	devicePanic(runMetalInt4Dequant(tensors[0], tensors[1]))
}

func (backend *Backend) Divergence1D(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	tensors := backend.tensorsAtPanic(input, output)
	spacingTensor := backend.uploadFloat32Scalar(spacing, format)
	devicePanic(runMetalPhysicsBinary(metalPhysicsDivergence1D, tensors[0], spacingTensor, tensors[1]))
}

func (backend *Backend) DoIntervene(adjacency, intervened, output unsafe.Pointer,
	nodeCount, intervenedCount int,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(adjacency, intervened, output)
	devicePanic(runMetalDoIntervene(tensors[0], tensors[1], tensors[2]))
}

func (backend *Backend) Dropout(dst, src unsafe.Pointer,
	count int,
	config device.DropoutConfig,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(src, dst)
	devicePanic(runMetalDropout(tensors[0], tensors[1]))
}

func (backend *Backend) ExpectedFreeEnergy(predictedObs, preferredObs, predictedState, output unsafe.Pointer,
	obsCount, stateCount int,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(predictedObs, preferredObs, predictedState, output)
	devicePanic(runMetalExpectedFreeEnergy(tensors[0], tensors[1], tensors[2], tensors[3]))
}

func (backend *Backend) FFT1D(realIn, imagIn, realOut, imagOut unsafe.Pointer, count int, format dtype.DType) {
	tensors := backend.tensorsAtPanic(realIn, imagIn, realOut, imagOut)
	devicePanic(runMetalFFT1D(tensors[0], tensors[1], tensors[2], tensors[3]))
}

func (backend *Backend) FlashAttention(config device.FlashAttentionConfig,
	query, key, value, output unsafe.Pointer,
	seqQ, seqK, depth, valueDim int,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(query, key, value, output)
	devicePanic(runMetalFlashAttention(tensors[0], tensors[1], tensors[2], tensors[3]))
}

func (backend *Backend) FreeEnergy(likelihood, posterior, prior, output unsafe.Pointer,
	count int,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(likelihood, posterior, prior, output)
	// The auxiliary tensor is used as scratch internally, but the API requires it to be passed
	// We can pass the output tensor as auxiliary since it will just be validated for DType and Bridge
	devicePanic(runMetalFreeEnergy(tensors[0], tensors[1], tensors[2], tensors[3], tensors[3]))
}

func (backend *Backend) FrontdoorAdjustment(mediatorGivenX, outcomeGivenXM, marginalX, output unsafe.Pointer,
	xCount, mediatorCount, yCount int,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(mediatorGivenX, outcomeGivenXM, marginalX, output)
	devicePanic(runMetalFrontdoorAdjustment(tensors[0], tensors[1], tensors[2], tensors[3]))
}

func (backend *Backend) Grad1D(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	tensors := backend.tensorsAtPanic(input, output)
	spacingTensor := backend.uploadFloat32Scalar(spacing, format)
	devicePanic(runMetalPhysicsBinary(metalPhysicsGrad1D, tensors[0], spacingTensor, tensors[1]))
}

func (backend *Backend) GroupNorm(config device.GroupNormConfig,
	input, scale, bias, output unsafe.Pointer,
	batch, channels, spatial int,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(input, scale, bias, output)
	devicePanic(runMetalGroupNorm(tensors[0], tensors[1], tensors[2], tensors[3]))
}

func (backend *Backend) HawkesIntensity(eventTimes, queryTimes, output unsafe.Pointer,
	eventCount, queryCount int,
	mu, alpha, beta float32,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(eventTimes, queryTimes, output)
	baseline := backend.uploadFloat32Scalar(mu, format)
	alphaTensor := backend.uploadFloat32Scalar(alpha, format)
	betaTensor := backend.uploadFloat32Scalar(beta, format)
	devicePanic(runMetalHawkesIntensity(tensors[0], tensors[1], baseline, alphaTensor, betaTensor, tensors[2]))
}

func (backend *Backend) HawkesKernelMatrix(eventTimes, output unsafe.Pointer,
	eventCount int,
	alpha, beta float32,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(eventTimes, output)
	alphaTensor := backend.uploadFloat32Scalar(alpha, format)
	betaTensor := backend.uploadFloat32Scalar(beta, format)
	devicePanic(runMetalHawkesKernelMatrix(tensors[0], alphaTensor, betaTensor, tensors[1]))
}

func (backend *Backend) HawkesLogLikelihood(eventTimes unsafe.Pointer,
	eventCount int,
	totalT, mu, alpha, beta float32,
	output unsafe.Pointer,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(eventTimes, output)
	totalTime := backend.uploadFloat32Scalar(totalT, format)
	baseline := backend.uploadFloat32Scalar(mu, format)
	alphaTensor := backend.uploadFloat32Scalar(alpha, format)
	betaTensor := backend.uploadFloat32Scalar(beta, format)
	devicePanic(runMetalHawkesLogLikelihood(tensors[0], totalTime, baseline, alphaTensor, betaTensor, tensors[1]))
}

func (backend *Backend) IFFT1D(realIn, imagIn, realOut, imagOut unsafe.Pointer, count int, format dtype.DType) {
	tensors := backend.tensorsAtPanic(realIn, imagIn, realOut, imagOut)
	devicePanic(runMetalIFFT1D(tensors[0], tensors[1], tensors[2], tensors[3]))
}

func (backend *Backend) IVEstimate(instrument, treatment, outcome unsafe.Pointer,
	count int,
	output unsafe.Pointer,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(instrument, treatment, outcome, output)
	devicePanic(runMetalIVEstimate(tensors[0], tensors[1], tensors[2], tensors[3]))
}

func (backend *Backend) InstanceNorm(input, scale, bias, output unsafe.Pointer,
	batch, channels, spatial int,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(input, scale, bias, output)
	devicePanic(runMetalInstanceNorm(tensors[0], tensors[1], tensors[2], tensors[3]))
}

func (backend *Backend) InversePermute(config device.VSAConfig, input, output unsafe.Pointer, count int, format dtype.DType) {
	tensors := backend.tensorsAtPanic(input, output)
	devicePanic(runMetalVSAInversePermuteKernel(tensors[0], tensors[1]))
}

func (backend *Backend) Laplacian(input, output unsafe.Pointer, dims []int, spacing float32, format dtype.DType) {
	tensors := backend.tensorsAtPanic(input, output)
	spacingTensor := backend.uploadFloat32Scalar(spacing, format)
	devicePanic(runMetalPhysicsBinary(metalPhysicsLaplacian, tensors[0], spacingTensor, tensors[1]))
}

func (backend *Backend) Laplacian4(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	tensors := backend.tensorsAtPanic(input, output)
	spacingTensor := backend.uploadFloat32Scalar(spacing, format)
	devicePanic(runMetalPhysicsBinary(metalPhysicsLaplacian4, tensors[0], spacingTensor, tensors[1]))
}

func (backend *Backend) MadelungContinuity(density, velocity, residual unsafe.Pointer,
	count int,
	spacing float32,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(density, velocity, residual)
	spacingTensor := backend.uploadFloat32Scalar(spacing, format)
	devicePanic(runMetalMadelungContinuity(tensors[0], tensors[1], spacingTensor, tensors[2]))
}

func (backend *Backend) MarkovBlanketPartition(adjacency, internal, output unsafe.Pointer,
	nodeCount, internalCount int,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(adjacency, internal, output)
	devicePanic(runMetalMarkovBlanketPartition(tensors[0], tensors[1], tensors[2]))
}

func (backend *Backend) MarkovFlowActive(mutualInformation, partition, output unsafe.Pointer,
	nodeCount int,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(mutualInformation, partition, output)
	devicePanic(runMetalMarkovFlow(tensors[0], tensors[1], tensors[2], 0))
}

func (backend *Backend) MarkovFlowInternal(mutualInformation, partition, output unsafe.Pointer,
	nodeCount int,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(mutualInformation, partition, output)
	devicePanic(runMetalMarkovFlow(tensors[0], tensors[1], tensors[2], 1))
}

func (backend *Backend) MarkovMutualInformation(joint, output unsafe.Pointer,
	xCount, yCount int,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(joint, output)
	devicePanic(runMetalMarkovMutualInformation(tensors[0], tensors[1]))
}

func (backend *Backend) MaxPool2D(config device.PoolConfig,
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(input, output)
	devicePanic(runMetalMaxPool2D(tensors[0], tensors[1]))
}

func (backend *Backend) MultiHeadAttention(config device.MultiHeadAttentionConfig,
	query, key, value, output unsafe.Pointer,
	seqQ, seqK int,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(query, key, value, output)
	devicePanic(runMetalMultiHeadAttention(tensors[0], tensors[1], tensors[2], tensors[3]))
}

func (backend *Backend) Permute(config device.VSAConfig, input, output unsafe.Pointer, count int, format dtype.DType) {
	tensors := backend.tensorsAtPanic(input, output)
	devicePanic(runMetalVSAPermuteKernel(tensors[0], tensors[1]))
}

func (backend *Backend) PrecisionWeight(errors, precision, output unsafe.Pointer, count int, format dtype.DType) {
	tensors := backend.tensorsAtPanic(errors, precision, output)
	devicePanic(runMetalPrecisionWeight(tensors[0], tensors[1], tensors[2]))
}

func (backend *Backend) Prediction(weights, representation, output unsafe.Pointer,
	outDim, inDim int,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(weights, representation, output)
	devicePanic(runMetalPCPrediction(tensors[0], tensors[1], tensors[2]))
}

func (backend *Backend) PredictionError(observed, predicted, output unsafe.Pointer,
	count int,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(observed, predicted, output)
	devicePanic(runMetalPCPredictionErrorKernel(tensors[0], tensors[1], tensors[2]))
}

func (backend *Backend) Quant(dst, src unsafe.Pointer, count int, config device.DequantInt8Config, dstFormat, srcFormat dtype.DType) {
	tensors := backend.tensorsAtPanic(src, dst)
	devicePanic(runMetalInt8Quant(tensors[0], tensors[1]))
}

func (backend *Backend) QuantumPotential(density, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	tensors := backend.tensorsAtPanic(density, output)
	spacingTensor := backend.uploadFloat32Scalar(spacing, format)
	devicePanic(runMetalPhysicsBinary(metalPhysicsQuantumPotential, tensors[0], spacingTensor, tensors[1]))
}

func (backend *Backend) RoPE(config device.RoPEConfig,
	input, output unsafe.Pointer,
	seqLen, numHeads, headDim int,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(input, output)
	devicePanic(runMetalRoPE(tensors[0], tensors[1]))
}

func (backend *Backend) RoPEPairs(output, input, cosBuffer, sinBuffer unsafe.Pointer,
	halfDim int,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(input, output)
	devicePanic(runMetalRoPE(tensors[0], tensors[1]))
}

func (backend *Backend) ScaledDotProductAttention(config device.FlashAttentionConfig,
	query, key, value, output unsafe.Pointer,
	seqQ, seqK, depth, valueDim int,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(query, key, value, output)
	devicePanic(runMetalAttention(tensors[0], tensors[1], tensors[2], tensors[3]))
}

func (backend *Backend) Similarity(left, right unsafe.Pointer, count int, format dtype.DType) float32 {
	tensors := backend.tensorsAtPanic(left, right)
	out := backend.emptyScalar(format)
	devicePanic(runMetalDot(tensors[0], tensors[1], out))
	return backend.readFloat32Scalar(out.residentPointer())
}

func (backend *Backend) UpdateRepresentation(config device.PredictiveCodingConfig,
	weights, representation, predictionError, output unsafe.Pointer,
	outDim, inDim int,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(weights, representation, predictionError, output)
	devicePanic(runMetalPCUpdateRepresentation(tensors[0], tensors[1], tensors[2], tensors[3]))
}

func (backend *Backend) UpdateWeights(config device.PredictiveCodingConfig,
	weights, representation, predictionError, output unsafe.Pointer,
	outDim, inDim int,
	format dtype.DType) {
	tensors := backend.tensorsAtPanic(weights, representation, predictionError, output)
	devicePanic(runMetalPCUpdateWeights(tensors[0], tensors[1], tensors[2], tensors[3]))
}
