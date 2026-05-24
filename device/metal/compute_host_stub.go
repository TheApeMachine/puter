//go:build !darwin || !cgo

package metal

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
	"github.com/theapemachine/puter/device/metal/activation"
	"github.com/theapemachine/puter/device/metal/elementwise"
	"github.com/theapemachine/puter/device/metal/losses"
	"github.com/theapemachine/puter/device/metal/reduction"
	"github.com/theapemachine/puter/device/metal/sampling"
)
func (host *ComputeHost) BinaryElementwise(dst, left, right unsafe.Pointer, format dtype.DType, kernel elementwise.BinaryKernel) {
	host.unavailable()
}

func (host *ComputeHost) DispatchALiBiBias(scores, slope, output unsafe.Pointer, seqQ, seqK int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchAdaptiveAvgPool2D(input, output unsafe.Pointer, batch, channels, inHeight, inWidth, outHeight, outWidth int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchAdaptiveMaxPool2D(input, output unsafe.Pointer, batch, channels, inHeight, inWidth, outHeight, outWidth int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchApplyMask(input, mask, output unsafe.Pointer, count int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchAvgPool2D(config device.PoolConfig, input, output unsafe.Pointer, batch, channels, inHeight, inWidth, outHeight, outWidth int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchAxpy(y, x unsafe.Pointer, alpha float32, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchBackdoorAdjustment(conditional, marginalZ, output unsafe.Pointer, xCount, zCount, yCount int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchBatchNormEval(input, scale, bias, mean, variance, output unsafe.Pointer, batch, channels, spatial int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchBeliefUpdate(likelihood, prior, output unsafe.Pointer, count int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchBind(left, right, output unsafe.Pointer, count int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchBohmianVelocity(phase, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchBundle(left, right, output unsafe.Pointer, count int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchCATE(treated, control, output unsafe.Pointer, count int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchCausalMask(output unsafe.Pointer, seqQ, seqK int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchCholesky(input, output unsafe.Pointer, matrixOrder int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchConv1D(config device.Conv1DConfig, input, weight, bias, output unsafe.Pointer, batch, inChannels, inLength, outChannels, kernelLength, outLength int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchConv2D(config device.Conv2DConfig, input, weight, bias, output unsafe.Pointer, batch, inChannels, inHeight, inWidth, outChannels, kernelHeight, kernelWidth, outHeight, outWidth int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchConv3D(config device.Conv3DConfig, input, weight, bias, output unsafe.Pointer, batch, inChannels, inD, inH, inW, outChannels, kD, kH, kW, outD, outH, outW int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchConvTranspose2D(config device.Conv2DConfig, input, weight, bias, output unsafe.Pointer, batch, inChannels, inHeight, inWidth, outChannels, kernelHeight, kernelWidth, outHeight, outWidth int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchCounterfactual(observedY, observedX, counterfactualX, output unsafe.Pointer, count int, slope float32, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchDAGMarkovFactorization(conditionals unsafe.Pointer, conditionalCount int, output unsafe.Pointer, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchDequant(dst, src unsafe.Pointer, count int, config device.DequantInt8Config, dstFormat, srcFormat dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchDequant4(dst, src unsafe.Pointer, pairCount int, config device.DequantInt4Config, dstFormat, srcFormat dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchDivergence1D(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchDoIntervene(adjacency, intervened, output unsafe.Pointer, nodeCount, intervenedCount int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchDropout(dst, src unsafe.Pointer, count int, config device.DropoutConfig, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchExpectedFreeEnergy(predictedObs, preferredObs, predictedState, output unsafe.Pointer, obsCount, stateCount int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchFFT1D(realIn, imagIn, realOut, imagOut unsafe.Pointer, count int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchFlashAttention(config device.FlashAttentionConfig, query, key, value, output unsafe.Pointer, seqQ, seqK, depth, valueDim int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchFreeEnergy(likelihood, posterior, prior, output unsafe.Pointer, count int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchFrontdoorAdjustment(mediatorGivenX, outcomeGivenXM, marginalX, output unsafe.Pointer, xCount, mediatorCount, yCount int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchGrad1D(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchGroupNorm(config device.GroupNormConfig, input, scale, bias, output unsafe.Pointer, batch, channels, spatial int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchHawkesIntensity(eventTimes, queryTimes, output unsafe.Pointer, eventCount, queryCount int, mu, alpha, beta float32, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchHawkesKernelMatrix(eventTimes, output unsafe.Pointer, eventCount int, alpha, beta float32, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchHawkesLogLikelihood(eventTimes unsafe.Pointer, eventCount int, totalT, mu, alpha, beta float32, output unsafe.Pointer, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchIFFT1D(realIn, imagIn, realOut, imagOut unsafe.Pointer, count int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchIVEstimate(instrument, treatment, outcome unsafe.Pointer, count int, output unsafe.Pointer, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchInstanceNorm(input, scale, bias, output unsafe.Pointer, batch, channels, spatial int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchInversePermute(config device.VSAConfig, input, output unsafe.Pointer, count int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchLaplacian(input, output unsafe.Pointer, dims []int, spacing float32, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchLaplacian4(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchMadelungContinuity(density, velocity, residual unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchMarkovBlanketPartition(adjacency, internal, output unsafe.Pointer, nodeCount, internalCount int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchMarkovFlowActive(mutualInformation, partition, output unsafe.Pointer, nodeCount int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchMarkovFlowInternal(mutualInformation, partition, output unsafe.Pointer, nodeCount int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchMarkovMutualInformation(joint, output unsafe.Pointer, xCount, yCount int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchMaxPool2D(config device.PoolConfig, input, output unsafe.Pointer, batch, channels, inHeight, inWidth, outHeight, outWidth int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchMultiHeadAttention(config device.MultiHeadAttentionConfig, query, key, value, output unsafe.Pointer, seqQ, seqK int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchPermute(config device.VSAConfig, input, output unsafe.Pointer, count int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchPrecisionWeight(errors, precision, output unsafe.Pointer, count int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchPrediction(weights, representation, output unsafe.Pointer, outDim, inDim int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchPredictionError(observed, predicted, output unsafe.Pointer, count int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchQuant(dst, src unsafe.Pointer, count int, config device.DequantInt8Config, dstFormat, srcFormat dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchQuantumPotential(density, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchRoPE(config device.RoPEConfig, input, output unsafe.Pointer, seqLen, numHeads, headDim int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchRoPEPairs(output, input, cosBuffer, sinBuffer unsafe.Pointer, halfDim int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchScaledDotProductAttention(config device.FlashAttentionConfig, query, key, value, output unsafe.Pointer, seqQ, seqK, depth, valueDim int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchUpdateRepresentation(config device.PredictiveCodingConfig, weights, representation, predictionError, output unsafe.Pointer, outDim, inDim int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchUpdateWeights(config device.PredictiveCodingConfig, weights, representation, predictionError, output unsafe.Pointer, outDim, inDim int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) GLUPacked(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType, variant activation.GLUVariant) {
	host.unavailable()
}

func (host *ComputeHost) GLUTensors(dst, gate, up unsafe.Pointer, format dtype.DType, variant activation.GLUVariant) {
	host.unavailable()
}

func (host *ComputeHost) LaunchBag(table, indices, offsets, output unsafe.Pointer, vocab, hidden, bagCount, indexCount int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) LaunchLayerNorm(input, scale, bias, output unsafe.Pointer, rows, lastDim int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) LaunchLookup(table, indices, output unsafe.Pointer, vocab, hidden, indexCount int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) LaunchRMSNorm(input, scale, output unsafe.Pointer, rows, lastDim int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) MatmulLaunch(out, left, right unsafe.Pointer, rows, inner, cols int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) PReLUV(dst, src, slopes unsafe.Pointer, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) Softmax(dst, src unsafe.Pointer, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) StandardUnary(dst, src unsafe.Pointer, format dtype.DType, kernel activation.StandardKernel) {
	host.unavailable()
}

func (host *ComputeHost) UnaryElementwise(dst, src unsafe.Pointer, format dtype.DType, kernel elementwise.UnaryKernel) {
	host.unavailable()
}

func (host *ComputeHost) UnaryParam(dst, src unsafe.Pointer, format dtype.DType, kernelName string, param float32) {
	host.unavailable()
}

func (host *ComputeHost) ReductionScalar(values unsafe.Pointer, count int, format dtype.DType, kernel reduction.ReductionKernel,) float32 {
	host.unavailable()
	return 0
}

func (host *ComputeHost) PairLossScalar(predictions, targets unsafe.Pointer, format dtype.DType, kernel losses.LossKernel,) float32 {
	host.unavailable()
	return 0
}

func (host *ComputeHost) CrossEntropyScalar(logits, targets unsafe.Pointer, batchSize, classes int, format dtype.DType,) float32 {
	host.unavailable()
	return 0
}

func (host *ComputeHost) SamplingIndex(kernel sampling.SamplingKernel, config device.SamplingConfig, logits unsafe.Pointer, vocabSize int, format dtype.DType,) int32 {
	host.unavailable()
	return 0
}

func (host *ComputeHost) DotProduct(left, right unsafe.Pointer, count int, format dtype.DType,) float32 {
	host.unavailable()
	return 0
}

func (host *ComputeHost) DispatchSimilarity(left, right unsafe.Pointer, count int, format dtype.DType,) float32 {
	host.unavailable()
	return 0
}

