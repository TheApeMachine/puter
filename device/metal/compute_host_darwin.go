//go:build darwin && cgo

package metal

import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device"
	cpuoptimizer "github.com/theapemachine/puter/device/cpu/optimizer"
	"github.com/theapemachine/puter/device/metal/activation"
	"github.com/theapemachine/puter/device/metal/attention"
	"github.com/theapemachine/puter/device/metal/checkpoint"
	"github.com/theapemachine/puter/device/metal/convolution"
	"github.com/theapemachine/puter/device/metal/dropout"
	"github.com/theapemachine/puter/device/metal/elementwise"
	"github.com/theapemachine/puter/device/metal/embedding"
	"github.com/theapemachine/puter/device/metal/layernorm"
	"github.com/theapemachine/puter/device/metal/masking"
	metalmath "github.com/theapemachine/puter/device/metal/math"
	"github.com/theapemachine/puter/device/metal/matmul"
	"github.com/theapemachine/puter/device/metal/model_editing"
	"github.com/theapemachine/puter/device/metal/normalization"
	metaloptimizer "github.com/theapemachine/puter/device/metal/optimizer"
	metalpool "github.com/theapemachine/puter/device/metal/pool"
	metalresonant "github.com/theapemachine/puter/device/metal/resonant"
	metalrope "github.com/theapemachine/puter/device/metal/rope"
	metalshape "github.com/theapemachine/puter/device/metal/shape"
)

type ComputeHost struct {
	bridge *metalBridge
}

var errInvalidModulatedLayerNormShape = errors.New("metal modulated layernorm: invalid shape")

func (host *ComputeHost) NeedsPlatform() {
	panic("metal: platform unavailable")
}

func (host *ComputeHost) unavailable() {
	panic("metal: dispatch not implemented")
}

func (host *ComputeHost) dispatchError(err error) {
	if err != nil {
		panic(err)
	}
}

func (host *ComputeHost) contextRef() uintptr {
	if host.bridge == nil {
		return 0
	}

	return host.bridge.contextRef()
}

func (host *ComputeHost) devicePointer() unsafe.Pointer {
	if host.bridge == nil {
		return nil
	}

	return unsafe.Pointer(host.bridge.device)
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

	shapeDims := deviceTensor.Shape().Dims()

	if len(shapeDims) == 0 {
		return 0, 0
	}

	cols = uint32(shapeDims[len(shapeDims)-1])
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

	if err := elementwise.DispatchBinaryElementwiseRefs(
		host.devicePointer(),
		unsafe.Pointer(resolveBufferRef(dst)),
		unsafe.Pointer(resolveBufferRef(left)),
		unsafe.Pointer(resolveBufferRef(right)),
		format,
		kernel,
		count,
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) DispatchAdaptiveAvgPool2D(input, output unsafe.Pointer, batch, channels, inHeight, inWidth, outHeight, outWidth int, format dtype.DType) {
	if batch == 0 || channels == 0 || inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 {
		return
	}

	host.dispatchError(metalpool.DispatchPool2DRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(input))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(batch),
		uint32(channels),
		uint32(inHeight),
		uint32(inWidth),
		uint32(outHeight),
		uint32(outWidth),
		false,
		true,
	))
}

func (host *ComputeHost) DispatchAdaptiveMaxPool2D(input, output unsafe.Pointer, batch, channels, inHeight, inWidth, outHeight, outWidth int, format dtype.DType) {
	if batch == 0 || channels == 0 || inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 {
		return
	}

	host.dispatchError(metalpool.DispatchPool2DRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(input))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(batch),
		uint32(channels),
		uint32(inHeight),
		uint32(inWidth),
		uint32(outHeight),
		uint32(outWidth),
		true,
		true,
	))
}

func (host *ComputeHost) DispatchAvgPool2D(config device.PoolConfig, input, output unsafe.Pointer, batch, channels, inHeight, inWidth, outHeight, outWidth int, format dtype.DType) {
	if batch == 0 || channels == 0 || inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 {
		return
	}

	host.dispatchError(metalpool.DispatchPool2DRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(input))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(batch),
		uint32(channels),
		uint32(inHeight),
		uint32(inWidth),
		uint32(outHeight),
		uint32(outWidth),
		false,
		false,
	))
}

func (host *ComputeHost) DispatchAxpy(y, x unsafe.Pointer, alpha float32, format dtype.DType) {
	count := host.elementCount(y, x)

	if count == 0 {
		return
	}

	if err := elementwise.DispatchAxpyRefs(
		host.devicePointer(),
		unsafe.Pointer(resolveBufferRef(y)),
		unsafe.Pointer(resolveBufferRef(x)),
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
	host.unavailable()
}

func (host *ComputeHost) DispatchBatchNormDenorm(input, mean, variance, output unsafe.Pointer, batch, channels, spatial int, format dtype.DType) {
	if batch == 0 || channels == 0 || spatial == 0 {
		return
	}

	if err := normalization.DispatchBatchNormDenormRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(input))),
		uintptr(unsafe.Pointer(resolveBufferRef(mean))),
		uintptr(unsafe.Pointer(resolveBufferRef(variance))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(batch*channels),
		uint32(channels),
		uint32(spatial),
	); err != nil {
		host.dispatchError(err)
	}
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

func (host *ComputeHost) DispatchCholesky(input, output unsafe.Pointer, matrixOrder int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchConv1D(config device.Conv1DConfig, input, weight, bias, output unsafe.Pointer, batch, inChannels, inLength, outChannels, kernelLength, outLength int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) DispatchConv2D(config device.Conv2DConfig, input, weight, bias, output unsafe.Pointer, batch, inChannels, inHeight, inWidth, outChannels, kernelHeight, kernelWidth, outHeight, outWidth int, format dtype.DType) {
	if batch == 0 || inChannels == 0 || inHeight == 0 || inWidth == 0 ||
		outChannels == 0 || kernelHeight == 0 || kernelWidth == 0 ||
		outHeight == 0 || outWidth == 0 {
		return
	}

	if config.StrideH <= 0 || config.StrideW <= 0 ||
		config.DilationH <= 0 || config.DilationW <= 0 {
		host.dispatchError(tensor.ErrShapeMismatch)
	}

	if err := convolution.DispatchConv2DRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(input))),
		uintptr(unsafe.Pointer(resolveBufferRef(weight))),
		uintptr(unsafe.Pointer(resolveBufferRef(bias))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(batch),
		uint32(inChannels),
		uint32(inHeight),
		uint32(inWidth),
		uint32(outChannels),
		uint32(kernelHeight),
		uint32(kernelWidth),
		uint32(outHeight),
		uint32(outWidth),
		uint32(config.StrideH),
		uint32(config.StrideW),
		uint32(config.PaddingH),
		uint32(config.PaddingW),
		uint32(config.DilationH),
		uint32(config.DilationW),
	); err != nil {
		host.dispatchError(err)
	}
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
	if count == 0 {
		return
	}

	if err := dropout.DispatchDropoutRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(src))),
		uintptr(unsafe.Pointer(resolveBufferRef(dst))),
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
	if batch == 0 || channels == 0 || spatial == 0 {
		return
	}

	if err := normalization.DispatchGroupNormRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(input))),
		uintptr(unsafe.Pointer(resolveBufferRef(scale))),
		uintptr(unsafe.Pointer(resolveBufferRef(bias))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(batch),
		uint32(channels),
		uint32(spatial),
		uint32(config.Groups),
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
	if batch == 0 || channels == 0 || inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 {
		return
	}

	host.dispatchError(metalpool.DispatchPool2DRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(input))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(batch),
		uint32(channels),
		uint32(inHeight),
		uint32(inWidth),
		uint32(outHeight),
		uint32(outWidth),
		true,
		false,
	))
}

func (host *ComputeHost) DispatchMultiHeadAttention(config device.MultiHeadAttentionConfig, query, key, value, output unsafe.Pointer, seqQ, seqK int, format dtype.DType) {
	if seqQ == 0 || seqK == 0 || config.NumHeads == 0 || config.HeadDim == 0 {
		return
	}

	kvHeads := config.KVHeadCount

	if kvHeads <= 0 {
		kvHeads = config.NumHeads
	}

	if err := validateMetalTensorBytes(query, "query", config.NumHeads*seqQ, config.HeadDim, format); err != nil {
		host.dispatchError(err)
		return
	}

	if err := validateMetalTensorBytes(key, "key", kvHeads*seqK, config.HeadDim, format); err != nil {
		host.dispatchError(err)
		return
	}

	if err := validateMetalTensorBytes(value, "value", kvHeads*seqK, config.HeadDim, format); err != nil {
		host.dispatchError(err)
		return
	}

	if err := validateMetalTensorBytes(output, "output", config.NumHeads*seqQ, config.HeadDim, format); err != nil {
		host.dispatchError(err)
		return
	}

	if err := attention.DispatchMultiHeadAttentionRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(query))),
		uintptr(unsafe.Pointer(resolveBufferRef(key))),
		uintptr(unsafe.Pointer(resolveBufferRef(value))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		config,
		uint32(seqQ),
		uint32(seqK),
		format,
	); err != nil {
		host.dispatchError(err)
	}
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
	if seqLen == 0 || numHeads == 0 || headDim == 0 {
		return
	}

	elementCount := seqLen * numHeads * headDim

	if err := validateDispatchPointer("rope input", input, elementCount, format); err != nil {
		host.dispatchError(err)
		return
	}

	if err := validateDispatchPointer("rope output", output, elementCount, format); err != nil {
		host.dispatchError(err)
		return
	}

	if err := metalrope.DispatchRoPERefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(input))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		config,
		uint32(seqLen),
		uint32(numHeads),
		uint32(headDim),
		format,
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) DispatchMultiAxisRoPE(
	config device.MultiAxisRoPEConfig,
	input, output unsafe.Pointer,
	batch, seqLen, numHeads, headDim int,
	format dtype.DType,
) {
	if batch == 0 || seqLen == 0 || numHeads == 0 || headDim == 0 || headDim%2 != 0 {
		return
	}

	if err := metalrope.DispatchMultiAxisRoPERefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(input))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		config,
		uint32(batch),
		uint32(seqLen),
		uint32(numHeads),
		uint32(headDim),
		format,
	); err != nil {
		host.dispatchError(err)
	}
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
	if batch == 0 || halfCount == 0 {
		return
	}

	if err := activation.DispatchGLUPackedRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(dst))),
		uintptr(unsafe.Pointer(resolveBufferRef(packed))),
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

	if err := activation.DispatchGLUTensorsRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(dst))),
		uintptr(unsafe.Pointer(resolveBufferRef(gate))),
		uintptr(unsafe.Pointer(resolveBufferRef(up))),
		format,
		variant,
		count,
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) LaunchBag(table, indices, offsets, output unsafe.Pointer, vocab, hidden, bagCount, indexCount int, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) LaunchLayerNorm(input, scale, bias, output unsafe.Pointer, rows, lastDim int, format dtype.DType) {
	if rows == 0 || lastDim == 0 {
		return
	}

	if err := layernorm.DispatchLayerNormRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(input))),
		uintptr(unsafe.Pointer(resolveBufferRef(scale))),
		uintptr(unsafe.Pointer(resolveBufferRef(bias))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(rows),
		uint32(lastDim),
	); err != nil {
		host.dispatchError(err)
	}
}

/*
LaunchLookup dispatches the embedding-table gather kernel on the active
Metal context. The dispatcher passes table/indices/output as unsafe
pointers to DeviceTensor structs (see metal.DeviceTensor.DispatchPointer
and puter/execution.pointerOf); resolveBufferRef walks each pointer back
to its MTLBuffer handle before handing them to the embedding bridge.

vocab/hidden/indexCount are the kernel's shape constants. indexCount ==
0 is a legal no-op (no tokens to embed); we short-circuit so the kernel
isn't launched with a zero grid.
*/
func (host *ComputeHost) LaunchLookup(table, indices, output unsafe.Pointer, vocab, hidden, indexCount int, format dtype.DType) {
	if indexCount == 0 || hidden == 0 || vocab == 0 {
		return
	}

	if err := validateMetalTensorBytes(table, "table", vocab, hidden, format); err != nil {
		host.dispatchError(err)
		return
	}

	if err := validateMetalTensorBytes(indices, "indices", indexCount, 1, dtype.Int32); err != nil {
		host.dispatchError(err)
		return
	}

	if err := validateMetalTensorBytes(output, "output", indexCount, hidden, format); err != nil {
		host.dispatchError(err)
		return
	}

	if err := embedding.DispatchLookupRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(table))),
		uintptr(unsafe.Pointer(resolveBufferRef(indices))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(vocab),
		uint32(hidden),
		uint32(indexCount),
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) LaunchTimestepEmbedding(
	config device.TimestepEmbeddingConfig,
	timesteps, output unsafe.Pointer,
	count, dim int,
	format dtype.DType,
) {
	if count == 0 || dim == 0 {
		return
	}

	host.dispatchError(config.Validate())

	if err := embedding.DispatchTimestepEmbeddingRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(timesteps))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		config.MaxPeriod,
		config.DownscaleFreqShift,
		config.TimestepDivisor,
		config.FlipSinToCos,
		uint32(count),
		uint32(dim),
	); err != nil {
		host.dispatchError(err)
	}
}

/*
LaunchRMSNorm runs the per-dtype RMSNorm kernel from
device/metal/layernorm/layer.metal. The dispatcher hands input/scale/out
as unsafe pointers to DeviceTensor structs; resolveBufferRef unwraps
them to MTLBuffer handles before the dispatch call.

The CPU contract is "rows × lastDim" — Metal's RMSNorm kernel agrees
(one threadgroup per row, threads sum across cols), so the two ints map
directly to the Metal call.
*/
func (host *ComputeHost) LaunchRMSNorm(
	config device.RMSNormConfig,
	input, scale, output unsafe.Pointer,
	rows, lastDim int,
	format dtype.DType,
) {
	if rows == 0 || lastDim == 0 {
		return
	}

	host.dispatchError(config.Validate())

	if err := validateMetalTensorBytes(input, "input", rows, lastDim, format); err != nil {
		host.dispatchError(err)
		return
	}

	if err := validateMetalTensorBytes(scale, "scale", 1, lastDim, format); err != nil {
		host.dispatchError(err)
		return
	}

	if err := validateMetalTensorBytes(output, "output", rows, lastDim, format); err != nil {
		host.dispatchError(err)
		return
	}

	if err := layernorm.DispatchRMSNormRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(input))),
		uintptr(unsafe.Pointer(resolveBufferRef(scale))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(rows),
		uint32(lastDim),
		float32(config.Epsilon),
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) LaunchAdaptiveRMSNorm(
	config device.RMSNormConfig,
	input, modulation, output unsafe.Pointer,
	rows, lastDim, rowsPerBatch, modulationCols int,
	format dtype.DType,
) {
	if rows == 0 || lastDim == 0 {
		return
	}

	host.dispatchError(config.Validate())

	if rowsPerBatch <= 0 || rows%rowsPerBatch != 0 {
		host.dispatchError(errInvalidModulatedLayerNormShape)
	}

	if modulationCols < 2*lastDim {
		host.dispatchError(errInvalidModulatedLayerNormShape)
	}

	if err := layernorm.DispatchAdaptiveRMSNormRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(input))),
		uintptr(unsafe.Pointer(resolveBufferRef(modulation))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(rows),
		uint32(lastDim),
		uint32(rowsPerBatch),
		uint32(modulationCols),
		float32(config.Epsilon),
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) LaunchModulatedLayerNorm(
	config device.ModulatedLayerNormConfig,
	input, modulation, output unsafe.Pointer,
	rows, lastDim, rowsPerBatch, modulationCols int,
	format dtype.DType,
) {
	if rows == 0 || lastDim == 0 {
		return
	}

	host.dispatchError(config.Validate())

	if rowsPerBatch <= 0 || rows%rowsPerBatch != 0 {
		host.dispatchError(errInvalidModulatedLayerNormShape)
	}

	if modulationCols < (config.Set*3+2)*lastDim {
		host.dispatchError(errInvalidModulatedLayerNormShape)
	}

	inputTensor := resolveDeviceTensor(input)
	modulationTensor := resolveDeviceTensor(modulation)

	if inputTensor != nil && modulationTensor != nil &&
		inputTensor.DType() != modulationTensor.DType() {
		host.dispatchError(fmt.Errorf(
			"metal modulated layernorm: input dtype %s does not match modulation dtype %s",
			inputTensor.DType(),
			modulationTensor.DType(),
		))

		return
	}

	if inputTensor != nil {
		format = inputTensor.DType()
	}

	host.dispatchError(validateDispatchPointer("modulated layernorm input", input, rows*lastDim, format))
	host.dispatchError(validateDispatchPointer(
		"modulated layernorm modulation",
		modulation,
		(rows/rowsPerBatch)*modulationCols,
		format,
	))
	host.dispatchError(validateDispatchPointer("modulated layernorm output", output, rows*lastDim, format))

	if err := normalization.DispatchModulatedLayerNormRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(input))),
		uintptr(unsafe.Pointer(resolveBufferRef(modulation))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(rows),
		uint32(lastDim),
		uint32(rowsPerBatch),
		uint32(modulationCols),
		uint32(config.Set),
		float32(config.Epsilon),
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) MatmulLaunch(out, left, right unsafe.Pointer, rows, inner, cols int, format dtype.DType) {
	if rows == 0 || inner == 0 || cols == 0 {
		return
	}

	if err := host.validateMatmulLaunch(out, left, right, rows, inner, cols, format); err != nil {
		host.dispatchError(err)
		return
	}

	if err := matmul.DispatchMatmulRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(left))),
		uintptr(unsafe.Pointer(resolveBufferRef(right))),
		uintptr(unsafe.Pointer(resolveBufferRef(out))),
		format,
		uint32(rows),
		uint32(inner),
		uint32(cols),
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) validateMatmulLaunch(
	out, left, right unsafe.Pointer,
	rows, inner, cols int,
	format dtype.DType,
) error {
	if rows < 0 || inner < 0 || cols < 0 {
		return fmt.Errorf("metal matmul: negative dimensions rows=%d inner=%d cols=%d", rows, inner, cols)
	}

	if err := validateMetalTensorBytes(left, "left", rows, inner, format); err != nil {
		return err
	}

	if err := validateMetalTensorBytes(right, "right", inner, cols, format); err != nil {
		return err
	}

	return validateMetalTensorBytes(out, "output", rows, cols, format)
}

func validateMetalTensorBytes(
	pointer unsafe.Pointer,
	name string,
	rows, cols int,
	format dtype.DType,
) error {
	deviceTensor := resolveDeviceTensor(pointer)

	if deviceTensor == nil {
		return fmt.Errorf("metal matmul: %s is not a resident Metal tensor", name)
	}

	if deviceTensor.DType() != format {
		return fmt.Errorf(
			"metal matmul: %s dtype %s does not match launch dtype %s",
			name, deviceTensor.DType(), format,
		)
	}

	requiredBytes, err := format.BytesFor(rows * cols)

	if err != nil {
		return fmt.Errorf("metal matmul: %s byte count: %w", name, err)
	}

	if deviceTensor.Bytes() < requiredBytes {
		return fmt.Errorf(
			"metal matmul: %s buffer has %d bytes, need %d bytes for %dx%d %s",
			name, deviceTensor.Bytes(), requiredBytes, rows, cols, format,
		)
	}

	return nil
}

func (host *ComputeHost) PReLUV(dst, src, slopes unsafe.Pointer, format dtype.DType) {
	host.unavailable()
}

func (host *ComputeHost) Softmax(dst, src unsafe.Pointer, format dtype.DType) {
	rows, cols := host.matrixRowsCols(src)

	if rows == 0 || cols == 0 {
		return
	}

	if err := activation.DispatchSoftmaxRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(dst))),
		uintptr(unsafe.Pointer(resolveBufferRef(src))),
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
	if err := activation.DispatchStandardUnaryRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(dst))),
		uintptr(unsafe.Pointer(resolveBufferRef(src))),
		format,
		kernel,
		count,
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) UnaryElementwise(dst, src unsafe.Pointer, format dtype.DType, kernel elementwise.UnaryKernel) {
	count := host.elementCount(dst, src)

	if count == 0 {
		return
	}

	if err := elementwise.DispatchUnaryMathRefs(
		host.devicePointer(),
		unsafe.Pointer(resolveBufferRef(dst)),
		unsafe.Pointer(resolveBufferRef(src)),
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

	if err := activation.DispatchUnaryParamRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(dst))),
		uintptr(unsafe.Pointer(resolveBufferRef(src))),
		format,
		kernelName,
		param,
		count,
	); err != nil {
		host.dispatchError(err)
	}
}
func (host *ComputeHost) DispatchApplyMask(input, mask, output unsafe.Pointer, count int, format dtype.DType) {
	if count == 0 {
		return
	}

	host.dispatchError(masking.DispatchApplyMaskRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(input))),
		uintptr(unsafe.Pointer(resolveBufferRef(mask))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(count),
	))
}

func (host *ComputeHost) DispatchCausalMask(output unsafe.Pointer, seqQ, seqK int, format dtype.DType) {
	if seqQ == 0 || seqK == 0 {
		return
	}

	host.dispatchError(masking.DispatchCausalMaskRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(seqQ),
		uint32(seqK),
	))
}

func (host *ComputeHost) DispatchALiBiBias(
	scores, slope, output unsafe.Pointer,
	seqQ, seqK int,
	format dtype.DType,
) {
	if seqQ == 0 || seqK == 0 {
		return
	}

	host.dispatchError(masking.DispatchALiBiBiasRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(scores))),
		uintptr(unsafe.Pointer(resolveBufferRef(slope))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(seqQ),
		uint32(seqK),
	))
}

func (host *ComputeHost) DispatchInvSqrtDimScale(
	out, input unsafe.Pointer,
	dim int32,
	format dtype.DType,
) {
	count := host.elementCount(input, out)

	if count == 0 || dim <= 0 || host.bridge == nil {
		return
	}

	dimBuffer := host.bridge.borrowScratch(4)

	defer host.bridge.releaseScratch(dimBuffer)

	host.bridge.writeInt32Scalar(dimBuffer, dim)

	host.dispatchError(metalmath.DispatchInvSqrtDimScaleRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(input))),
		uintptr(unsafe.Pointer(dimBuffer)),
		uintptr(unsafe.Pointer(resolveBufferRef(out))),
		format,
		uint32(count),
	))
}

func (host *ComputeHost) DispatchLogSumExp(input, output unsafe.Pointer, cols int, format dtype.DType) {
	rows, columnCount := host.matrixRowsCols(input)

	if rows == 0 || columnCount == 0 {
		return
	}

	if cols > 0 {
		columnCount = uint32(cols)
	}

	host.dispatchError(metalmath.DispatchLogSumExpRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(input))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		rows,
		columnCount,
	))
}

func (host *ComputeHost) DispatchOuter(
	left, right, output unsafe.Pointer,
	leftCount, rightCount int,
	format dtype.DType,
) {
	if leftCount == 0 || rightCount == 0 {
		return
	}

	host.dispatchError(metalmath.DispatchOuterRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(left))),
		uintptr(unsafe.Pointer(resolveBufferRef(right))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(leftCount),
		uint32(rightCount),
	))
}

func (host *ComputeHost) DispatchWeightGraftAdd(
	weights, injection unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	if count == 0 {
		return
	}

	host.dispatchError(model_editing.DispatchWeightGraftAddRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(weights))),
		uintptr(unsafe.Pointer(resolveBufferRef(injection))),
		format,
		uint32(count),
	))
}

func (host *ComputeHost) DispatchCheckpointEncode(input, output unsafe.Pointer, format dtype.DType) {
	inputTensor := resolveDeviceTensor(input)
	outputTensor := resolveDeviceTensor(output)

	if inputTensor == nil || outputTensor == nil {
		host.dispatchError(tensor.ErrNeedsPlatformSetup)
	}

	elementCount, err := checkpoint.CheckpointElementCount(format, inputTensor.Len())

	if err != nil {
		host.dispatchError(err)
	}

	inputDims := inputTensor.Shape().Dims()
	dims := make([]uint64, len(inputDims))

	for index, dimension := range inputDims {
		dims[index] = uint64(dimension)
	}

	host.dispatchError(checkpoint.DispatchCheckpointEncodeRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(input))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		uint32(len(inputDims)),
		elementCount,
		dims,
	))
}

func (host *ComputeHost) DispatchCheckpointDecode(input, output unsafe.Pointer, format dtype.DType) {
	inputTensor := resolveDeviceTensor(input)
	outputTensor := resolveDeviceTensor(output)

	if inputTensor == nil || outputTensor == nil {
		host.dispatchError(tensor.ErrNeedsPlatformSetup)
	}

	elementCount, err := checkpoint.CheckpointElementCount(format, outputTensor.Len())

	if err != nil {
		host.dispatchError(err)
	}

	host.dispatchError(checkpoint.DispatchCheckpointDecodeRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(input))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		checkpoint.CheckpointHeaderBytes(outputTensor.Shape().Rank()),
		elementCount,
	))
}

func (host *ComputeHost) DispatchAdagrad(
	config cpuoptimizer.AdagradConfig,
	params, gradients, accumulator, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	if count == 0 {
		return
	}

	host.dispatchError(metaloptimizer.DispatchOptimizer3Refs(
		host.contextRef(),
		metaloptimizer.OperationAdagrad,
		uintptr(unsafe.Pointer(resolveBufferRef(params))),
		uintptr(unsafe.Pointer(resolveBufferRef(gradients))),
		uintptr(unsafe.Pointer(resolveBufferRef(accumulator))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(count),
		metaloptimizer.AdagradMetalConfig(config),
	))
}

func (host *ComputeHost) DispatchAdam(
	config cpuoptimizer.AdamConfig,
	params, gradients, firstMoment, secondMoment, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	if count == 0 {
		return
	}

	host.dispatchError(metaloptimizer.DispatchOptimizer4Refs(
		host.contextRef(),
		metaloptimizer.OperationAdam,
		uintptr(unsafe.Pointer(resolveBufferRef(params))),
		uintptr(unsafe.Pointer(resolveBufferRef(gradients))),
		uintptr(unsafe.Pointer(resolveBufferRef(firstMoment))),
		uintptr(unsafe.Pointer(resolveBufferRef(secondMoment))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(count),
		metaloptimizer.AdamMetalConfig(config),
	))
}

func (host *ComputeHost) DispatchAdamax(
	config cpuoptimizer.AdamaxConfig,
	params, gradients, firstMoment, infinityMoment, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	if count == 0 {
		return
	}

	host.dispatchError(metaloptimizer.DispatchOptimizer4Refs(
		host.contextRef(),
		metaloptimizer.OperationAdamax,
		uintptr(unsafe.Pointer(resolveBufferRef(params))),
		uintptr(unsafe.Pointer(resolveBufferRef(gradients))),
		uintptr(unsafe.Pointer(resolveBufferRef(firstMoment))),
		uintptr(unsafe.Pointer(resolveBufferRef(infinityMoment))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(count),
		metaloptimizer.AdamaxMetalConfig(config),
	))
}

func (host *ComputeHost) DispatchAdamW(
	config cpuoptimizer.AdamWConfig,
	params, gradients, firstMoment, secondMoment, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	if count == 0 {
		return
	}

	host.dispatchError(metaloptimizer.DispatchOptimizer4Refs(
		host.contextRef(),
		metaloptimizer.OperationAdamW,
		uintptr(unsafe.Pointer(resolveBufferRef(params))),
		uintptr(unsafe.Pointer(resolveBufferRef(gradients))),
		uintptr(unsafe.Pointer(resolveBufferRef(firstMoment))),
		uintptr(unsafe.Pointer(resolveBufferRef(secondMoment))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(count),
		metaloptimizer.AdamWMetalConfig(config),
	))
}

func (host *ComputeHost) DispatchHebbian(
	config cpuoptimizer.HebbianConfig,
	weights, post, pre, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	postTensor := resolveDeviceTensor(post)
	preTensor := resolveDeviceTensor(pre)

	if postTensor == nil || preTensor == nil || count == 0 {
		return
	}

	postDims := postTensor.Shape().Dims()
	preDims := preTensor.Shape().Dims()
	postCount := uint32(1)
	preCount := uint32(1)

	for _, dimension := range postDims {
		postCount *= uint32(dimension)
	}

	for _, dimension := range preDims {
		preCount *= uint32(dimension)
	}

	host.dispatchError(metaloptimizer.DispatchHebbianRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(weights))),
		uintptr(unsafe.Pointer(resolveBufferRef(post))),
		uintptr(unsafe.Pointer(resolveBufferRef(pre))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		postCount,
		preCount,
		config,
	))
}

func (host *ComputeHost) DispatchLARS(
	config cpuoptimizer.LARSConfig,
	params, gradients, momentum, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	if count == 0 {
		return
	}

	host.dispatchError(metaloptimizer.DispatchLARSRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(params))),
		uintptr(unsafe.Pointer(resolveBufferRef(gradients))),
		uintptr(unsafe.Pointer(resolveBufferRef(momentum))),
		uintptr(unsafe.Pointer(resolveBufferRef(momentum))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(count),
		1,
		config,
	))
}

func (host *ComputeHost) DispatchLBFGS(
	config cpuoptimizer.LBFGSConfig,
	params, gradients, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	if count == 0 {
		return
	}

	host.dispatchError(metaloptimizer.DispatchOptimizer2Refs(
		host.contextRef(),
		metaloptimizer.OperationLBFGS,
		uintptr(unsafe.Pointer(resolveBufferRef(params))),
		uintptr(unsafe.Pointer(resolveBufferRef(gradients))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(count),
		metaloptimizer.LBFGSMetalConfig(config),
	))
}

func (host *ComputeHost) DispatchLion(
	config cpuoptimizer.LionConfig,
	params, gradients, momentum, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	if count == 0 {
		return
	}

	host.dispatchError(metaloptimizer.DispatchOptimizer3Refs(
		host.contextRef(),
		metaloptimizer.OperationLion,
		uintptr(unsafe.Pointer(resolveBufferRef(params))),
		uintptr(unsafe.Pointer(resolveBufferRef(gradients))),
		uintptr(unsafe.Pointer(resolveBufferRef(momentum))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(count),
		metaloptimizer.LionMetalConfig(config),
	))
}

func (host *ComputeHost) DispatchRMSprop(
	config cpuoptimizer.RMSpropConfig,
	params, gradients, secondMoment, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	if count == 0 {
		return
	}

	host.dispatchError(metaloptimizer.DispatchOptimizer3Refs(
		host.contextRef(),
		metaloptimizer.OperationRMSprop,
		uintptr(unsafe.Pointer(resolveBufferRef(params))),
		uintptr(unsafe.Pointer(resolveBufferRef(gradients))),
		uintptr(unsafe.Pointer(resolveBufferRef(secondMoment))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(count),
		metaloptimizer.RMSpropMetalConfig(config),
	))
}

func (host *ComputeHost) DispatchSGD(
	config cpuoptimizer.SGDConfig,
	params, gradients, momentum, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	if count == 0 {
		return
	}

	host.dispatchError(metaloptimizer.DispatchOptimizer3Refs(
		host.contextRef(),
		metaloptimizer.OperationSGD,
		uintptr(unsafe.Pointer(resolveBufferRef(params))),
		uintptr(unsafe.Pointer(resolveBufferRef(gradients))),
		uintptr(unsafe.Pointer(resolveBufferRef(momentum))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(count),
		metaloptimizer.SGDMetalConfig(config),
	))
}

func (host *ComputeHost) DispatchCopyContiguous(dst, src unsafe.Pointer, count int, format dtype.DType) {
	elementBytes := metalshape.ElementByteSize(format)

	if count == 0 || elementBytes == 0 {
		return
	}

	host.dispatchError(metalshape.DispatchCopyBytesRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(src))),
		uintptr(unsafe.Pointer(resolveBufferRef(dst))),
		format,
		uint32(count*elementBytes),
	))
}

func (host *ComputeHost) DispatchResonantUpdateForward(
	x, y, vr, vi, diag unsafe.Pointer,
	xOut, yOut, aOut, bOut, invROut unsafe.Pointer,
	batchTime, headCount, headDim int,
	config device.ResonantUpdateConfig,
	format dtype.DType,
) {
	if batchTime*headCount*headDim == 0 {
		return
	}

	if err := metalresonant.DispatchResonantUpdateForwardRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(x))),
		uintptr(unsafe.Pointer(resolveBufferRef(y))),
		uintptr(unsafe.Pointer(resolveBufferRef(vr))),
		uintptr(unsafe.Pointer(resolveBufferRef(vi))),
		uintptr(unsafe.Pointer(resolveBufferRef(diag))),
		uintptr(unsafe.Pointer(resolveBufferRef(xOut))),
		uintptr(unsafe.Pointer(resolveBufferRef(yOut))),
		uintptr(unsafe.Pointer(resolveBufferRef(aOut))),
		uintptr(unsafe.Pointer(resolveBufferRef(bOut))),
		uintptr(unsafe.Pointer(resolveBufferRef(invROut))),
		batchTime,
		headCount,
		headDim,
		config,
		format,
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) DispatchResonantUpdateBackward(
	gradXOut, gradYOut unsafe.Pointer,
	x, y, diag, a, b, invR unsafe.Pointer,
	gradX, gradY, gradVR, gradVI unsafe.Pointer,
	batchTime, headCount, headDim int,
	config device.ResonantUpdateConfig,
	format dtype.DType,
) {
	if batchTime*headCount*headDim == 0 {
		return
	}

	if err := metalresonant.DispatchResonantUpdateBackwardRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(gradXOut))),
		uintptr(unsafe.Pointer(resolveBufferRef(gradYOut))),
		uintptr(unsafe.Pointer(resolveBufferRef(x))),
		uintptr(unsafe.Pointer(resolveBufferRef(y))),
		uintptr(unsafe.Pointer(resolveBufferRef(diag))),
		uintptr(unsafe.Pointer(resolveBufferRef(a))),
		uintptr(unsafe.Pointer(resolveBufferRef(b))),
		uintptr(unsafe.Pointer(resolveBufferRef(invR))),
		uintptr(unsafe.Pointer(resolveBufferRef(gradX))),
		uintptr(unsafe.Pointer(resolveBufferRef(gradY))),
		uintptr(unsafe.Pointer(resolveBufferRef(gradVR))),
		uintptr(unsafe.Pointer(resolveBufferRef(gradVI))),
		batchTime,
		headCount,
		headDim,
		config,
		format,
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) DispatchReshape(input, output unsafe.Pointer, count int, format dtype.DType) {
	host.DispatchCopyContiguous(output, input, count, format)
}

func (host *ComputeHost) DispatchSplitHeads(
	input, output unsafe.Pointer,
	batch, seq, heads, headDim int,
	format dtype.DType,
) {
	host.DispatchCopyContiguous(output, input, batch*seq*heads*headDim, format)
}

func (host *ComputeHost) DispatchViewAsHeads(
	input, output unsafe.Pointer,
	batch, seq, numHeads, headDim int,
	format dtype.DType,
) {
	host.DispatchCopyContiguous(output, input, batch*seq*numHeads*headDim, format)
}

func (host *ComputeHost) DispatchConcat(left, right, output unsafe.Pointer, format dtype.DType) {
	leftTensor := resolveDeviceTensor(left)
	rightTensor := resolveDeviceTensor(right)

	if leftTensor == nil || rightTensor == nil {
		host.dispatchError(tensor.ErrNeedsPlatformSetup)
	}

	elementBytes := metalshape.ElementByteSize(format)

	if elementBytes == 0 {
		host.dispatchError(tensor.ErrNeedsPlatformSetup)
	}

	leftBytes := uint32(leftTensor.Bytes())
	rightBytes := uint32(rightTensor.Bytes())

	host.dispatchError(metalshape.DispatchConcatBytesRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(left))),
		uintptr(unsafe.Pointer(resolveBufferRef(right))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		leftBytes,
		rightBytes,
	))
}

func (host *ComputeHost) DispatchSplit2(input, left, right unsafe.Pointer, format dtype.DType) {
	leftTensor := resolveDeviceTensor(left)
	rightTensor := resolveDeviceTensor(right)

	if leftTensor == nil || rightTensor == nil {
		host.dispatchError(tensor.ErrNeedsPlatformSetup)
	}

	elementBytes := metalshape.ElementByteSize(format)

	if elementBytes == 0 {
		host.dispatchError(tensor.ErrNeedsPlatformSetup)
	}

	leftBytes := uint32(leftTensor.Bytes())
	rightBytes := uint32(rightTensor.Bytes())

	host.dispatchError(metalshape.DispatchSplit2BytesRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(input))),
		uintptr(unsafe.Pointer(resolveBufferRef(left))),
		uintptr(unsafe.Pointer(resolveBufferRef(right))),
		format,
		leftBytes,
		rightBytes,
	))
}

func (host *ComputeHost) DispatchGather(
	source, indices, output unsafe.Pointer,
	outerDim, innerDim int,
	format dtype.DType,
) {
	outputTensor := resolveDeviceTensor(output)

	if outputTensor == nil || outerDim == 0 || innerDim == 0 {
		return
	}

	outRows := uint32(outputTensor.Len() / innerDim)

	host.dispatchError(metalshape.DispatchGatherRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(source))),
		uintptr(unsafe.Pointer(resolveBufferRef(indices))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(outerDim),
		uint32(innerDim),
		outRows,
	))
}

func (host *ComputeHost) DispatchScatter(
	target, indices, updates, output unsafe.Pointer,
	outerDim, innerDim int,
	format dtype.DType,
) {
	updatesTensor := resolveDeviceTensor(updates)

	if updatesTensor == nil || outerDim == 0 || innerDim == 0 {
		return
	}

	updateRows := uint32(updatesTensor.Len() / innerDim)

	host.dispatchError(metalshape.DispatchScatterRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(target))),
		uintptr(unsafe.Pointer(resolveBufferRef(indices))),
		uintptr(unsafe.Pointer(resolveBufferRef(updates))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(outerDim),
		uint32(innerDim),
		updateRows,
	))
}

func (host *ComputeHost) DispatchMergeHeads(
	input, output unsafe.Pointer,
	batch, seq, heads, headDim int,
	format dtype.DType,
) {
	if batch == 0 || seq == 0 || heads == 0 || headDim == 0 {
		return
	}

	host.dispatchError(metalshape.DispatchMergeHeadsRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(input))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(batch),
		uint32(seq),
		uint32(heads),
		uint32(headDim),
	))
}

func (host *ComputeHost) DispatchTranspose2D(
	input, output unsafe.Pointer,
	rows, cols int,
	format dtype.DType,
) {
	if rows == 0 || cols == 0 {
		return
	}

	host.dispatchError(metalshape.DispatchTranspose2DBytesRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(input))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(rows),
		uint32(cols),
	))
}

func (host *ComputeHost) DispatchLastToken(
	input, output unsafe.Pointer,
	batch, seq, hidden int,
	format dtype.DType,
) {
	elementBytes := metalshape.ElementByteSize(format)

	if batch == 0 || seq == 0 || hidden == 0 || elementBytes == 0 {
		return
	}

	hiddenBytes := uint32(hidden * elementBytes)
	outBytes := uint32(batch * hidden * elementBytes)

	host.dispatchError(metalshape.DispatchLastTokenBytesRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(input))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(seq),
		hiddenBytes,
		outBytes,
	))
}

func (host *ComputeHost) DispatchUpsampleNearest2D(
	input, output unsafe.Pointer,
	batch, channels, inHeight, inWidth, outHeight, outWidth int,
	format dtype.DType,
) {
	if batch == 0 || channels == 0 || inHeight == 0 || inWidth == 0 || outHeight == 0 || outWidth == 0 {
		return
	}

	outputTensor := resolveDeviceTensor(output)
	outElements := uint32(outputTensor.Len() / batch)

	host.dispatchError(metalshape.DispatchUpsampleNearest2DBytesRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(input))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(channels),
		uint32(inHeight),
		uint32(inWidth),
		uint32(outHeight),
		uint32(outWidth),
		outElements,
	))
}

func (host *ComputeHost) DispatchWhere(
	mask, positive, negative, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	if count == 0 {
		return
	}

	host.dispatchError(metalshape.DispatchWhereRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(mask))),
		uintptr(unsafe.Pointer(resolveBufferRef(positive))),
		uintptr(unsafe.Pointer(resolveBufferRef(negative))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(count),
	))
}

func (host *ComputeHost) DispatchMaskedFill(
	input, mask, fill, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	if count == 0 {
		return
	}

	host.dispatchError(metalshape.DispatchMaskedFillRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(input))),
		uintptr(unsafe.Pointer(resolveBufferRef(mask))),
		uintptr(unsafe.Pointer(resolveBufferRef(fill))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(count),
	))
}

func (host *ComputeHost) DispatchSlice(
	input, output unsafe.Pointer,
	dim, start, end int,
	format dtype.DType,
) {
	inputTensor := resolveDeviceTensor(input)
	outputTensor := resolveDeviceTensor(output)
	elementBytes := metalshape.ElementByteSize(format)

	if inputTensor == nil || outputTensor == nil || elementBytes == 0 || dim < 0 || end <= start {
		host.dispatchError(tensor.ErrShapeMismatch)
	}

	inputDims := inputTensor.Shape().Dims()

	if dim >= len(inputDims) {
		host.dispatchError(tensor.ErrShapeMismatch)
	}

	sliceLen := end - start
	inner := 1

	for index := dim + 1; index < len(inputDims); index++ {
		inner *= inputDims[index]
	}

	innerBytes := uint32(inner * elementBytes)

	host.dispatchError(metalshape.DispatchSliceBytesRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(input))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(sliceLen),
		uint32(inputDims[dim]),
		innerBytes,
		uint32(start),
		uint32(outputTensor.Bytes()),
	))
}

func (host *ComputeHost) DispatchTranspose(
	input, permutation, output unsafe.Pointer,
	rank int,
	format dtype.DType,
) {
	inputTensor := resolveDeviceTensor(input)

	if inputTensor == nil || rank <= 0 {
		return
	}

	permutationView := unsafe.Slice((*int32)(permutation), rank)
	permutationValues := make([]uint32, rank)
	inputStrides := make([]uint32, rank)
	outputStrides := make([]uint32, rank)
	inputDims := inputTensor.Shape().Dims()

	if len(inputDims) != rank {
		host.dispatchError(tensor.ErrShapeMismatch)
	}

	stride := 1

	for index := rank - 1; index >= 0; index-- {
		inputStrides[index] = uint32(stride)
		stride *= inputDims[index]
	}

	outputDims := make([]int, rank)

	for index := 0; index < rank; index++ {
		permutationValues[index] = uint32(permutationView[index])
		outputDims[index] = inputDims[permutationView[index]]
	}

	stride = 1

	for index := rank - 1; index >= 0; index-- {
		outputStrides[index] = uint32(stride)
		stride *= outputDims[index]
	}

	host.dispatchError(metalshape.DispatchTransposeRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(input))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(rank),
		uint32(inputTensor.Len()),
		permutationValues,
		inputStrides,
		outputStrides,
	))
}

func (host *ComputeHost) DispatchPageWrite(
	storage, values, pageIDs, offsets, output unsafe.Pointer,
	pageSize int,
	format dtype.DType,
) {
	storageTensor := resolveDeviceTensor(storage)
	valuesTensor := resolveDeviceTensor(values)

	if storageTensor == nil || valuesTensor == nil || pageSize <= 0 {
		host.dispatchError(tensor.ErrShapeMismatch)
	}

	storageDims := storageTensor.Shape().Dims()
	valueDims := valuesTensor.Shape().Dims()

	if len(storageDims) < 2 || len(valueDims) != len(storageDims)-1 || storageDims[1] != pageSize {
		host.dispatchError(tensor.ErrShapeMismatch)
	}

	inner := 1

	for index := 2; index < len(storageDims); index++ {
		inner *= storageDims[index]
	}

	host.dispatchError(metalshape.DispatchPageWriteRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(storage))),
		uintptr(unsafe.Pointer(resolveBufferRef(values))),
		uintptr(unsafe.Pointer(resolveBufferRef(pageIDs))),
		uintptr(unsafe.Pointer(resolveBufferRef(offsets))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(storageDims[0]),
		uint32(pageSize),
		uint32(inner),
		uint32(valueDims[0]),
		0,
		0,
	))
}

func (host *ComputeHost) DispatchPageGather(
	storage, pageTable, pageSize, output unsafe.Pointer,
	format dtype.DType,
) {
	host.dispatchPageGather(storage, pageTable, pageSize, output, 0, format)
}

func (host *ComputeHost) DispatchPageGatherWithLiveRows(
	storage, pageTable, pageSize, output unsafe.Pointer,
	liveRows int,
	format dtype.DType,
) {
	host.dispatchPageGather(storage, pageTable, pageSize, output, liveRows, format)
}

func (host *ComputeHost) dispatchPageGather(
	storage, pageTable, pageSizePointer, output unsafe.Pointer,
	liveRows int,
	format dtype.DType,
) {
	storageTensor := resolveDeviceTensor(storage)
	pageTableTensor := resolveDeviceTensor(pageTable)
	outputTensor := resolveDeviceTensor(output)
	pageSize := host.int32FromDeviceScalar(pageSizePointer)

	if storageTensor == nil || pageTableTensor == nil || outputTensor == nil || pageSize <= 0 {
		host.dispatchError(tensor.ErrShapeMismatch)
	}

	storageDims := storageTensor.Shape().Dims()
	outputDims := outputTensor.Shape().Dims()

	if len(storageDims) < 2 || len(outputDims) != len(storageDims)-1 || storageDims[1] != pageSize {
		host.dispatchError(tensor.ErrShapeMismatch)
	}

	inner := 1

	for index := 2; index < len(storageDims); index++ {
		inner *= storageDims[index]
	}

	outRows := outputDims[0]

	if liveRows > 0 && liveRows < outRows {
		outRows = liveRows
	}

	maxRows := pageTableTensor.Len() * pageSize

	if maxRows < outRows {
		outRows = maxRows
	}

	host.dispatchError(metalshape.DispatchPageGatherRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(storage))),
		uintptr(unsafe.Pointer(resolveBufferRef(pageTable))),
		uintptr(unsafe.Pointer(resolveBufferRef(output))),
		format,
		uint32(storageDims[0]),
		uint32(pageSize),
		uint32(inner),
		uint32(outRows),
		0,
		0,
	))
}

func (host *ComputeHost) int32FromDeviceScalar(pointer unsafe.Pointer) int {
	if host.bridge == nil {
		return 0
	}

	return int(host.bridge.readInt32Scalar(resolveBufferRef(pointer)))
}
