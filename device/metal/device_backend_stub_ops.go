//go:build !darwin || !cgo

package metal

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

func (backend *Backend) Dot(left, right unsafe.Pointer, count int, format dtype.DType) float32 {
	backend.deviceNeedsPlatform()
	return 0
}

func (backend *Backend) Matmul(out, left, right unsafe.Pointer, rows, inner, cols int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) MaxPool2D(config device.PoolConfig, input, output unsafe.Pointer, batch, channels, inHeight, inWidth, outHeight, outWidth int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) AvgPool2D(config device.PoolConfig, input, output unsafe.Pointer, batch, channels, inHeight, inWidth, outHeight, outWidth int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) AdaptiveMaxPool2D(input, output unsafe.Pointer, batch, channels, inHeight, inWidth, outHeight, outWidth int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) AdaptiveAvgPool2D(input, output unsafe.Pointer, batch, channels, inHeight, inWidth, outHeight, outWidth int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Conv2D(config device.Conv2DConfig, input, weight, bias, output unsafe.Pointer, batch, inChannels, inHeight, inWidth, outChannels, kernelHeight, kernelWidth, outHeight, outWidth int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Conv1D(config device.Conv1DConfig, input, weight, bias, output unsafe.Pointer, batch, inChannels, inLength, outChannels, kernelLength, outLength int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Conv3D(config device.Conv3DConfig, input, weight, bias, output unsafe.Pointer, batch, inChannels, inD, inH, inW, outChannels, kD, kH, kW, outD, outH, outW int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) ConvTranspose2D(config device.Conv2DConfig, input, weight, bias, output unsafe.Pointer, batch, inChannels, inHeight, inWidth, outChannels, kernelHeight, kernelWidth, outHeight, outWidth int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Dropout(dst, src unsafe.Pointer, count int, config device.DropoutConfig, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) MSE(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32 {
	backend.deviceNeedsPlatform()
	return 0
}

func (backend *Backend) MAE(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32 {
	backend.deviceNeedsPlatform()
	return 0
}

func (backend *Backend) Huber(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32 {
	backend.deviceNeedsPlatform()
	return 0
}

func (backend *Backend) BinaryCrossEntropy(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32 {
	backend.deviceNeedsPlatform()
	return 0
}

func (backend *Backend) KLDivergence(predictions, targets unsafe.Pointer, count int, format dtype.DType) float32 {
	backend.deviceNeedsPlatform()
	return 0
}

func (backend *Backend) CrossEntropy(logits unsafe.Pointer, targets unsafe.Pointer, batchSize, classes int, format dtype.DType) float32 {
	backend.deviceNeedsPlatform()
	return 0
}

func (backend *Backend) GreedySample(logits unsafe.Pointer, vocabSize int, format dtype.DType) int32 {
	backend.deviceNeedsPlatform()
	return 0
}

func (backend *Backend) TopKSample(config device.SamplingConfig, logits unsafe.Pointer, vocabSize int, format dtype.DType) int32 {
	backend.deviceNeedsPlatform()
	return 0
}

func (backend *Backend) TopPSample(config device.SamplingConfig, logits unsafe.Pointer, vocabSize int, format dtype.DType) int32 {
	backend.deviceNeedsPlatform()
	return 0
}

func (backend *Backend) Lookup(table, indices, output unsafe.Pointer, vocab, hidden, indexCount int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Bag(table, indices, offsets, output unsafe.Pointer, vocab, hidden, bagCount, indexCount int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) GroupNorm(config device.GroupNormConfig, input, scale, bias, output unsafe.Pointer, batch, channels, spatial int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) InstanceNorm(input, scale, bias, output unsafe.Pointer, batch, channels, spatial int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) BatchNormEval(input, scale, bias, mean, variance, output unsafe.Pointer, batch, channels, spatial int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) LayerNorm(input, scale, bias, output unsafe.Pointer, rows, lastDim int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) RMSNorm(input, scale, output unsafe.Pointer, rows, lastDim int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) RoPE(config device.RoPEConfig, input, output unsafe.Pointer, seqLen, numHeads, headDim int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) RoPEPairs(output, input, cosBuffer, sinBuffer unsafe.Pointer, halfDim int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) HawkesIntensity(eventTimes, queryTimes, output unsafe.Pointer, eventCount, queryCount int, mu, alpha, beta float32, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) HawkesKernelMatrix(eventTimes, output unsafe.Pointer, eventCount int, alpha, beta float32, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) HawkesLogLikelihood(eventTimes unsafe.Pointer, eventCount int, totalT, mu, alpha, beta float32, output unsafe.Pointer, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) MarkovMutualInformation(joint, output unsafe.Pointer, xCount, yCount int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) MarkovBlanketPartition(adjacency, internal, output unsafe.Pointer, nodeCount, internalCount int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Laplacian(input, output unsafe.Pointer, dims []int, spacing float32, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Laplacian4(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Grad1D(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Divergence1D(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) FFT1D(realIn, imagIn, realOut, imagOut unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) IFFT1D(realIn, imagIn, realOut, imagOut unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) QuantumPotential(density, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) BohmianVelocity(phase, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) MadelungContinuity(density, velocity, residual unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Cholesky(input, output unsafe.Pointer, matrixOrder int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) BackdoorAdjustment(conditional, marginalZ, output unsafe.Pointer, xCount, zCount, yCount int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) FrontdoorAdjustment(mediatorGivenX, outcomeGivenXM, marginalX, output unsafe.Pointer, xCount, mediatorCount, yCount int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) DoIntervene(adjacency, intervened, output unsafe.Pointer, nodeCount, intervenedCount int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) CATE(treated, control, output unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Counterfactual(observedY, observedX, counterfactualX, output unsafe.Pointer, count int, slope float32, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) IVEstimate(instrument, treatment, outcome unsafe.Pointer, count int, output unsafe.Pointer, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) DAGMarkovFactorization(conditionals unsafe.Pointer, conditionalCount int, output unsafe.Pointer, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) MarkovFlowActive(mutualInformation, partition, output unsafe.Pointer, nodeCount int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) MarkovFlowInternal(mutualInformation, partition, output unsafe.Pointer, nodeCount int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) ScaledDotProductAttention(config device.FlashAttentionConfig, query, key, value, output unsafe.Pointer, seqQ, seqK, depth, valueDim int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) FlashAttention(config device.FlashAttentionConfig, query, key, value, output unsafe.Pointer, seqQ, seqK, depth, valueDim int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) MultiHeadAttention(config device.MultiHeadAttentionConfig, query, key, value, output unsafe.Pointer, seqQ, seqK int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Bind(left, right, output unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Bundle(left, right, output unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Permute(config device.VSAConfig, input, output unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) InversePermute(config device.VSAConfig, input, output unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Similarity(left, right unsafe.Pointer, count int, format dtype.DType) float32 {
	backend.deviceNeedsPlatform()
	return 0
}

func (backend *Backend) FreeEnergy(likelihood, posterior, prior, output unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) ExpectedFreeEnergy(predictedObs, preferredObs, predictedState, output unsafe.Pointer, obsCount, stateCount int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) BeliefUpdate(likelihood, prior, output unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) PrecisionWeight(errors, precision, output unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Prediction(weights, representation, output unsafe.Pointer, outDim, inDim int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) PredictionError(observed, predicted, output unsafe.Pointer, count int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) UpdateRepresentation(config device.PredictiveCodingConfig, weights, representation, predictionError, output unsafe.Pointer, outDim, inDim int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) UpdateWeights(config device.PredictiveCodingConfig, weights, representation, predictionError, output unsafe.Pointer, outDim, inDim int, format dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Dequant(dst, src unsafe.Pointer, count int, config device.DequantInt8Config, dstFormat, srcFormat dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Dequant4(dst, src unsafe.Pointer, pairCount int, config device.DequantInt4Config, dstFormat, srcFormat dtype.DType) {
	backend.deviceNeedsPlatform()
}

func (backend *Backend) Quant(dst, src unsafe.Pointer, count int, config device.DequantInt8Config, dstFormat, srcFormat dtype.DType) {
	backend.deviceNeedsPlatform()
}
