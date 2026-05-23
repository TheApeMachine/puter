//go:build cuda

package cuda

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
	"github.com/theapemachine/puter/device/cuda/activation"
	"github.com/theapemachine/puter/device/cuda/attention"
	"github.com/theapemachine/puter/device/cuda/convolution"
	"github.com/theapemachine/puter/device/cuda/dropout"
	"github.com/theapemachine/puter/device/cuda/elementwise"
	"github.com/theapemachine/puter/device/cuda/embedding"
	"github.com/theapemachine/puter/device/cuda/layernorm"
	"github.com/theapemachine/puter/device/cuda/matmul"
	"github.com/theapemachine/puter/device/cuda/normalization"
	"github.com/theapemachine/puter/device/cuda/pool"
	"github.com/theapemachine/puter/device/cuda/quant"
	"github.com/theapemachine/puter/device/cuda/rope"
	"github.com/theapemachine/puter/device/cuda/vsa"
)

type ComputeHost struct {
	bridge *cudaBridge
}

func (host *ComputeHost) NeedsPlatform() {
	panic("cuda: platform unavailable")
}

func (host *ComputeHost) unavailable() {
	panic("cuda: dispatch not implemented")
}

func (host *ComputeHost) dispatchError(err error) {
	if err != nil {
		panic(err)
	}
}

func (host *ComputeHost) contextRef() C.CUDADeviceRef {
	if host.bridge == nil {
		return nil
	}
	return host.bridge.contextRef()
}

func (host *ComputeHost) elementCount(pointers ...unsafe.Pointer) uint32 {
	for _, pointer := range pointers {
		deviceTensor := resolveDeviceTensor(pointer)
		if deviceTensor != nil {
			return uint32(deviceTensor.Len())
		}
	}
	return 0
}

func (host *ComputeHost) matrixRowsCols(pointer unsafe.Pointer) (rows uint32, cols uint32) {
	deviceTensor := resolveDeviceTensor(pointer)

	if deviceTensor == nil {
		return 0, 0
	}

	shape := deviceTensor.Shape()

	if len(shape) == 0 {
		return 0, 0
	}

	cols = uint32(shape[len(shape)-1])
	total := uint32(deviceTensor.Len())

	if cols == 0 {
		return 0, 0
	}

	return total / cols, cols
}

func (host *ComputeHost) BinaryElementwise(dst, left, right unsafe.Pointer, format dtype.DType, kernel elementwise.BinaryKernel) {
	count := host.elementCount(dst, left, right)

	if count == 0 {
		return
	}

	if err := elementwise.DispatchBinaryElementwise(
		host.contextRef(),
		resolveBufferRef(dst),
		resolveBufferRef(left),
		resolveBufferRef(right),
		format,
		kernel,
		count,
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) DispatchALiBiBias(scores, slope, output unsafe.Pointer, seqQ, seqK int, format dtype.DType) {
	if seqQ == 0 || seqK == 0 {
		return
	}

	if err := attention.DispatchALiBiBias(
		host.contextRef(),
		resolveBufferRef(scores),
		resolveBufferRef(slope),
		resolveBufferRef(output),
		uint32(seqQ),
		uint32(seqK),
		format,
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) DispatchAdaptiveAvgPool2D(input, output unsafe.Pointer, batch, channels, inHeight, inWidth, outHeight, outWidth int, format dtype.DType) {
	if batch == 0 || channels == 0 || inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 {
		return
	}

	if err := pool.DispatchAdaptiveAvgPool2D(
		host.contextRef(),
		resolveBufferRef(input),
		resolveBufferRef(output),
		format,
		uint32(batch),
		uint32(channels),
		uint32(inHeight),
		uint32(inWidth),
		uint32(outHeight),
		uint32(outWidth),
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) DispatchAdaptiveMaxPool2D(input, output unsafe.Pointer, batch, channels, inHeight, inWidth, outHeight, outWidth int, format dtype.DType) {
	if batch == 0 || channels == 0 || inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 {
		return
	}

	if err := pool.DispatchAdaptiveMaxPool2D(
		host.contextRef(),
		resolveBufferRef(input),
		resolveBufferRef(output),
		format,
		uint32(batch),
		uint32(channels),
		uint32(inHeight),
		uint32(inWidth),
		uint32(outHeight),
		uint32(outWidth),
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) DispatchApplyMask(input, mask, output unsafe.Pointer, count int, format dtype.DType) {
	if count == 0 {
		return
	}

	if err := attention.DispatchApplyMask(
		host.contextRef(),
		resolveBufferRef(input),
		resolveBufferRef(mask),
		resolveBufferRef(output),
		uint32(count),
		format,
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) DispatchAvgPool2D(config device.PoolConfig, input, output unsafe.Pointer, batch, channels, inHeight, inWidth, outHeight, outWidth int, format dtype.DType) {
	_ = config

	if batch == 0 || channels == 0 || inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 {
		return
	}

	if err := pool.DispatchAvgPool2D(
		host.contextRef(),
		resolveBufferRef(input),
		resolveBufferRef(output),
		format,
		uint32(batch),
		uint32(channels),
		uint32(inHeight),
		uint32(inWidth),
		uint32(outHeight),
		uint32(outWidth),
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) DispatchAxpy(y, x unsafe.Pointer, alpha float32, format dtype.DType) {
	count := host.elementCount(y, x)

	if count == 0 {
		return
	}

	if err := elementwise.DispatchAxpy(
		host.contextRef(),
		resolveBufferRef(y),
		resolveBufferRef(x),
		format,
		alpha,
		count,
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) DispatchBackdoorAdjustment(conditional, marginalZ, output unsafe.Pointer, xCount, zCount, yCount int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchBatchNormEval(input, scale, bias, mean, variance, output unsafe.Pointer, batch, channels, spatial int, format dtype.DType) {
	if batch == 0 || channels == 0 || spatial == 0 {
		return
	}

	if err := normalization.DispatchBatchNormEval(
		host.contextRef(),
		resolveBufferRef(input),
		resolveBufferRef(scale),
		resolveBufferRef(bias),
		resolveBufferRef(mean),
		resolveBufferRef(variance),
		resolveBufferRef(output),
		uint32(batch),
		uint32(channels),
		uint32(spatial),
		format,
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) DispatchBeliefUpdate(likelihood, prior, output unsafe.Pointer, count int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchBind(left, right, output unsafe.Pointer, count int, format dtype.DType) {
	elementCount := host.elementCount(left, right, output)

	if elementCount == 0 {
		return
	}

	if err := vsa.DispatchBind(
		host.contextRef(),
		resolveBufferRef(left),
		resolveBufferRef(right),
		resolveBufferRef(output),
		format,
		elementCount,
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) DispatchBohmianVelocity(phase, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchBundle(left, right, output unsafe.Pointer, count int, format dtype.DType) {
	elementCount := host.elementCount(left, right, output)

	if elementCount == 0 {
		return
	}

	if err := vsa.DispatchBundle(
		host.contextRef(),
		resolveBufferRef(left),
		resolveBufferRef(right),
		resolveBufferRef(output),
		format,
		elementCount,
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) DispatchCATE(treated, control, output unsafe.Pointer, count int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchCausalMask(output unsafe.Pointer, seqQ, seqK int, format dtype.DType) {
	if seqQ == 0 || seqK == 0 {
		return
	}

	if err := attention.DispatchCausalMask(
		host.contextRef(),
		resolveBufferRef(output),
		uint32(seqQ),
		uint32(seqK),
		format,
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) DispatchCholesky(input, output unsafe.Pointer, matrixOrder int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchConv1D(config device.Conv1DConfig, input, weight, bias, output unsafe.Pointer, batch, inChannels, inLength, outChannels, kernelLength, outLength int, format dtype.DType) {
	_ = config

	if batch == 0 || inChannels == 0 || outChannels == 0 || kernelLength == 0 {
		return
	}

	if err := convolution.DispatchConv1D(
		host.contextRef(),
		resolveBufferRef(input),
		resolveBufferRef(weight),
		resolveBufferRef(bias),
		resolveBufferRef(output),
		uint32(batch),
		uint32(inChannels),
		uint32(inLength),
		uint32(outChannels),
		uint32(kernelLength),
		uint32(outLength),
		format,
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) DispatchConv2D(_ device.Conv2DConfig, input, weight, bias, output unsafe.Pointer, batch, inChannels, inHeight, inWidth, outChannels, kernelHeight, kernelWidth, outHeight, outWidth int, format dtype.DType) {
	if batch == 0 || inChannels == 0 || outChannels == 0 || kernelHeight == 0 || kernelWidth == 0 {
		return
	}

	if err := convolution.DispatchConv2D(
		host.contextRef(),
		resolveBufferRef(input),
		resolveBufferRef(weight),
		resolveBufferRef(bias),
		resolveBufferRef(output),
		uint32(batch),
		uint32(inChannels),
		uint32(inHeight),
		uint32(inWidth),
		uint32(outChannels),
		uint32(kernelHeight),
		uint32(kernelWidth),
		uint32(outHeight),
		uint32(outWidth),
		format,
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) DispatchConv3D(config device.Conv3DConfig, input, weight, bias, output unsafe.Pointer, batch, inChannels, inD, inH, inW, outChannels, kD, kH, kW, outD, outH, outW int, format dtype.DType) {
	_ = config

	if batch == 0 || inChannels == 0 || outChannels == 0 || kD == 0 || kH == 0 || kW == 0 {
		return
	}

	if err := convolution.DispatchConv3D(
		host.contextRef(),
		resolveBufferRef(input),
		resolveBufferRef(weight),
		resolveBufferRef(bias),
		resolveBufferRef(output),
		uint32(batch),
		uint32(inChannels),
		uint32(inD),
		uint32(inH),
		uint32(inW),
		uint32(outChannels),
		uint32(kD),
		uint32(kH),
		uint32(kW),
		uint32(outD),
		uint32(outH),
		uint32(outW),
		format,
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) DispatchConvTranspose2D(config device.Conv2DConfig, input, weight, bias, output unsafe.Pointer, batch, inChannels, inHeight, inWidth, outChannels, kernelHeight, kernelWidth, outHeight, outWidth int, format dtype.DType) {
	_ = config

	if batch == 0 || inChannels == 0 || outChannels == 0 || kernelHeight == 0 || kernelWidth == 0 {
		return
	}

	if err := convolution.DispatchConvTranspose2D(
		host.contextRef(),
		resolveBufferRef(input),
		resolveBufferRef(weight),
		resolveBufferRef(bias),
		resolveBufferRef(output),
		uint32(batch),
		uint32(inChannels),
		uint32(inHeight),
		uint32(inWidth),
		uint32(outChannels),
		uint32(kernelHeight),
		uint32(kernelWidth),
		uint32(outHeight),
		uint32(outWidth),
		format,
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) DispatchCounterfactual(observedY, observedX, counterfactualX, output unsafe.Pointer, count int, slope float32, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchDAGMarkovFactorization(conditionals unsafe.Pointer, conditionalCount int, output unsafe.Pointer, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchDequant(dst, src unsafe.Pointer, count int, config device.DequantInt8Config, dstFormat, srcFormat dtype.DType) {
	_ = dstFormat
	_ = srcFormat

	if count == 0 {
		return
	}

	if err := quant.DispatchDequant(
		host.contextRef(),
		resolveBufferRef(src),
		resolveBufferRef(dst),
		config.Scale,
		config.ZeroPoint,
		uint32(count),
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) DispatchDequant4(dst, src unsafe.Pointer, pairCount int, config device.DequantInt4Config, dstFormat, srcFormat dtype.DType) {
	_ = dstFormat
	_ = srcFormat

	if pairCount == 0 {
		return
	}

	if err := quant.DispatchDequant4(
		host.contextRef(),
		resolveBufferRef(src),
		resolveBufferRef(dst),
		config.Scale,
		config.ZeroPoint,
		uint32(pairCount),
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) DispatchDivergence1D(input, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchDoIntervene(adjacency, intervened, output unsafe.Pointer, nodeCount, intervenedCount int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchDropout(dst, src unsafe.Pointer, count int, config device.DropoutConfig, format dtype.DType) {
	if count == 0 {
		return
	}

	if err := dropout.DispatchDropout(
		host.contextRef(),
		resolveBufferRef(src),
		resolveBufferRef(dst),
		uint32(count),
		config,
		format,
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) DispatchExpectedFreeEnergy(predictedObs, preferredObs, predictedState, output unsafe.Pointer, obsCount, stateCount int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchFFT1D(realIn, imagIn, realOut, imagOut unsafe.Pointer, count int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchFlashAttention(config device.FlashAttentionConfig, query, key, value, output unsafe.Pointer, seqQ, seqK, depth, valueDim int, format dtype.DType) {
	_ = config

	if seqQ == 0 || seqK == 0 || depth == 0 || valueDim == 0 {
		return
	}

	if err := attention.DispatchFlashAttention(
		host.contextRef(),
		resolveBufferRef(query),
		resolveBufferRef(key),
		resolveBufferRef(value),
		resolveBufferRef(output),
		uint32(seqQ),
		uint32(seqK),
		uint32(depth),
		uint32(valueDim),
		format,
	); err != nil {
		host.dispatchError(err)
	}
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
	if batch == 0 || channels == 0 || spatial == 0 || config.Groups == 0 {
		return
	}

	if err := normalization.DispatchGroupNorm(
		host.contextRef(),
		resolveBufferRef(input),
		resolveBufferRef(scale),
		resolveBufferRef(bias),
		resolveBufferRef(output),
		config,
		uint32(batch),
		uint32(channels),
		uint32(spatial),
		format,
	); err != nil {
		host.dispatchError(err)
	}
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
	if batch == 0 || channels == 0 || spatial == 0 {
		return
	}

	if err := normalization.DispatchInstanceNorm(
		host.contextRef(),
		resolveBufferRef(input),
		resolveBufferRef(scale),
		resolveBufferRef(bias),
		resolveBufferRef(output),
		uint32(batch),
		uint32(channels),
		uint32(spatial),
		format,
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) DispatchInversePermute(config device.VSAConfig, input, output unsafe.Pointer, count int, format dtype.DType) {
	_ = config

	elementCount := host.elementCount(input, output)

	if elementCount == 0 {
		return
	}

	if err := vsa.DispatchInversePermute(
		host.contextRef(),
		resolveBufferRef(input),
		resolveBufferRef(output),
		format,
		elementCount,
	); err != nil {
		host.dispatchError(err)
	}
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
	_ = config

	if batch == 0 || channels == 0 || inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 {
		return
	}

	if err := pool.DispatchMaxPool2D(
		host.contextRef(),
		resolveBufferRef(input),
		resolveBufferRef(output),
		format,
		uint32(batch),
		uint32(channels),
		uint32(inHeight),
		uint32(inWidth),
		uint32(outHeight),
		uint32(outWidth),
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) DispatchMultiHeadAttention(config device.MultiHeadAttentionConfig, query, key, value, output unsafe.Pointer, seqQ, seqK int, format dtype.DType) {
	if seqQ == 0 || seqK == 0 || config.NumHeads == 0 || config.HeadDim == 0 {
		return
	}

	if err := attention.DispatchMultiHeadAttention(
		host.contextRef(),
		resolveBufferRef(query),
		resolveBufferRef(key),
		resolveBufferRef(value),
		resolveBufferRef(output),
		config,
		uint32(seqQ),
		uint32(seqK),
		format,
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) DispatchPermute(config device.VSAConfig, input, output unsafe.Pointer, count int, format dtype.DType) {
	_ = config

	elementCount := host.elementCount(input, output)

	if elementCount == 0 {
		return
	}

	if err := vsa.DispatchPermute(
		host.contextRef(),
		resolveBufferRef(input),
		resolveBufferRef(output),
		format,
		elementCount,
	); err != nil {
		host.dispatchError(err)
	}
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
	_ = dstFormat
	_ = srcFormat

	if count == 0 {
		return
	}

	if err := quant.DispatchQuant(
		host.contextRef(),
		resolveBufferRef(src),
		resolveBufferRef(dst),
		config.Scale,
		config.ZeroPoint,
		uint32(count),
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) DispatchQuantumPotential(density, output unsafe.Pointer, count int, spacing float32, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchRoPE(config device.RoPEConfig, input, output unsafe.Pointer, seqLen, numHeads, headDim int, format dtype.DType) {
	if seqLen == 0 || numHeads == 0 || headDim == 0 || headDim%2 != 0 {
		return
	}

	if err := rope.DispatchRoPE(
		host.contextRef(),
		resolveBufferRef(input),
		resolveBufferRef(output),
		config,
		uint32(seqLen),
		uint32(numHeads),
		uint32(headDim),
		format,
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) DispatchRoPEPairs(output, input, cosBuffer, sinBuffer unsafe.Pointer, halfDim int, format dtype.DType) {
	if halfDim == 0 {
		return
	}

	if err := rope.DispatchRoPEPairs(
		host.contextRef(),
		resolveBufferRef(input),
		resolveBufferRef(output),
		resolveBufferRef(cosBuffer),
		resolveBufferRef(sinBuffer),
		uint32(halfDim),
		format,
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) DispatchScaledDotProductAttention(config device.FlashAttentionConfig, query, key, value, output unsafe.Pointer, seqQ, seqK, depth, valueDim int, format dtype.DType) {
	_ = config

	if seqQ == 0 || seqK == 0 || depth == 0 || valueDim == 0 {
		return
	}

	scratchBytes := attentionScoresBytes(seqQ, seqK)
	scoresBuffer := host.bridge.borrowScratch(scratchBytes)

	defer host.bridge.releaseScratch(scoresBuffer)

	if err := attention.DispatchScaledDotProductAttention(
		host.contextRef(),
		resolveBufferRef(query),
		resolveBufferRef(key),
		resolveBufferRef(value),
		scoresBuffer,
		resolveBufferRef(output),
		uint32(seqQ),
		uint32(seqK),
		uint32(depth),
		uint32(valueDim),
		format,
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) DispatchUpdateRepresentation(config device.PredictiveCodingConfig, weights, representation, predictionError, output unsafe.Pointer, outDim, inDim int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchUpdateWeights(config device.PredictiveCodingConfig, weights, representation, predictionError, output unsafe.Pointer, outDim, inDim int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) GLUPacked(dst, packed unsafe.Pointer, batch, halfCount int, format dtype.DType, variant activation.GLUVariant) {
	if batch == 0 || halfCount == 0 {
		return
	}

	if err := activation.DispatchGLUPacked(
		host.contextRef(),
		resolveBufferRef(dst),
		resolveBufferRef(packed),
		format,
		variant,
		uint32(halfCount),
		uint32(batch),
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) GLUTensors(dst, gate, up unsafe.Pointer, format dtype.DType, variant activation.GLUVariant) {
	count := host.elementCount(dst, gate, up)

	if count == 0 {
		return
	}

	if err := activation.DispatchGLUTensors(
		host.contextRef(),
		resolveBufferRef(dst),
		resolveBufferRef(gate),
		resolveBufferRef(up),
		format,
		variant,
		count,
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) LaunchBag(table, indices, offsets, output unsafe.Pointer, vocab, hidden, bagCount, indexCount int, format dtype.DType) {
	if vocab == 0 || hidden == 0 || bagCount == 0 || indexCount == 0 {
		return
	}

	if err := embedding.DispatchBag(
		host.contextRef(),
		resolveBufferRef(table),
		resolveBufferRef(indices),
		resolveBufferRef(offsets),
		resolveBufferRef(output),
		format,
		uint32(vocab),
		uint32(hidden),
		uint32(indexCount),
		uint32(bagCount),
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) LaunchLayerNorm(input, scale, bias, output unsafe.Pointer, rows, lastDim int, format dtype.DType) {
	if rows == 0 || lastDim == 0 {
		return
	}

	if err := layernorm.DispatchLayerNorm(
		host.contextRef(),
		resolveBufferRef(input),
		resolveBufferRef(scale),
		resolveBufferRef(bias),
		resolveBufferRef(output),
		format,
		uint32(rows),
		uint32(lastDim),
		0,
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) LaunchLookup(table, indices, output unsafe.Pointer, vocab, hidden, indexCount int, format dtype.DType) {
	if vocab == 0 || hidden == 0 || indexCount == 0 {
		return
	}

	if err := embedding.DispatchLookup(
		host.contextRef(),
		resolveBufferRef(table),
		resolveBufferRef(indices),
		resolveBufferRef(output),
		nil,
		format,
		uint32(vocab),
		uint32(hidden),
		uint32(indexCount),
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) LaunchRMSNorm(input, scale, output unsafe.Pointer, rows, lastDim int, format dtype.DType) {
	if rows == 0 || lastDim == 0 {
		return
	}

	if err := layernorm.DispatchRMSNorm(
		host.contextRef(),
		resolveBufferRef(input),
		resolveBufferRef(scale),
		resolveBufferRef(output),
		format,
		uint32(rows),
		uint32(lastDim),
		0,
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) MatmulLaunch(out, left, right unsafe.Pointer, rows, inner, cols int, format dtype.DType) {
	if rows == 0 || inner == 0 || cols == 0 {
		return
	}

	if err := matmul.DispatchMatmul(
		host.contextRef(),
		resolveBufferRef(left),
		resolveBufferRef(right),
		resolveBufferRef(out),
		format,
		uint32(rows),
		uint32(inner),
		uint32(cols),
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) PReLUV(dst, src, slopes unsafe.Pointer, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) Softmax(dst, src unsafe.Pointer, format dtype.DType) {
	rows, cols := host.matrixRowsCols(src)

	if rows == 0 || cols == 0 {
		return
	}

	if err := activation.DispatchSoftmax(
		host.contextRef(),
		resolveBufferRef(dst),
		resolveBufferRef(src),
		format,
		rows,
		cols,
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) StandardUnary(dst, src unsafe.Pointer, format dtype.DType, kernel activation.StandardKernel) {
	count := host.elementCount(dst, src)
	if count == 0 {
		return
	}
	if err := activation.DispatchStandardUnary(host.contextRef(), resolveBufferRef(dst), resolveBufferRef(src), format, kernel, count); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) UnaryElementwise(dst, src unsafe.Pointer, format dtype.DType, kernel elementwise.UnaryKernel) {
	count := host.elementCount(dst, src)

	if count == 0 {
		return
	}

	if err := elementwise.DispatchUnaryMath(
		host.contextRef(),
		resolveBufferRef(dst),
		resolveBufferRef(src),
		format,
		kernel,
		count,
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) UnaryParam(dst, src unsafe.Pointer, format dtype.DType, kernelName string, param float32) {
	count := host.elementCount(dst, src)

	if count == 0 {
		return
	}

	if err := activation.DispatchUnaryParam(
		host.contextRef(),
		resolveBufferRef(dst),
		resolveBufferRef(src),
		format,
		kernelName,
		param,
		count,
	); err != nil {
		host.dispatchError(err)
	}
}
