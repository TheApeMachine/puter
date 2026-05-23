//go:build cuda

package cuda

import (
	"math/rand/v2"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
	"github.com/theapemachine/puter/device/cuda/dot"
	"github.com/theapemachine/puter/device/cuda/losses"
	"github.com/theapemachine/puter/device/cuda/reduction"
	"github.com/theapemachine/puter/device/cuda/sampling"
)

func (host *ComputeHost) ReductionScalar(
	values unsafe.Pointer,
	count int,
	format dtype.DType,
	kernel reduction.ReductionKernel,
) float32 {
	if count == 0 || host.bridge == nil {
		return 0
	}

	elementCount := uint32(count)
	scratchA := host.bridge.borrowScratch(reductionScratchBytes(elementCount))
	scratchB := host.bridge.borrowScratch(reductionScratchBytes(elementCount))
	outBuffer := host.bridge.borrowScratch(4)

	defer host.bridge.releaseScratch(scratchA)
	defer host.bridge.releaseScratch(scratchB)
	defer host.bridge.releaseScratch(outBuffer)

	if err := reduction.DispatchReduction(
		host.contextRef(),
		resolveBufferRef(values),
		scratchA,
		scratchB,
		outBuffer,
		format,
		kernel,
		elementCount,
	); err != nil {
		host.dispatchError(err)
	}

	return host.bridge.readFloat32Scalar(outBuffer)
}

func (host *ComputeHost) DotProduct(
	left, right unsafe.Pointer,
	count int,
	format dtype.DType,
) float32 {
	if count == 0 || host.bridge == nil {
		return 0
	}

	elementCount := uint32(count)
	scratchBuffer := host.bridge.borrowScratch(dotScratchBytes(elementCount))
	outBuffer := host.bridge.borrowScratch(4)

	defer host.bridge.releaseScratch(scratchBuffer)
	defer host.bridge.releaseScratch(outBuffer)

	if err := dot.DispatchDot(
		host.contextRef(),
		resolveBufferRef(left),
		resolveBufferRef(right),
		scratchBuffer,
		outBuffer,
		format,
		elementCount,
	); err != nil {
		host.dispatchError(err)
	}

	return host.bridge.readFloat32Scalar(outBuffer)
}

func (host *ComputeHost) PairLossScalar(
	predictions, targets unsafe.Pointer,
	format dtype.DType,
	kernel losses.LossKernel,
) float32 {
	count := host.elementCount(predictions, targets)

	if count == 0 || host.bridge == nil {
		return 0
	}

	scratchBuffer := host.bridge.borrowScratch(reductionScratchBytes(count))
	outBuffer := host.bridge.borrowScratch(4)

	defer host.bridge.releaseScratch(scratchBuffer)
	defer host.bridge.releaseScratch(outBuffer)

	if err := losses.DispatchPairLoss(
		host.contextRef(),
		resolveBufferRef(predictions),
		resolveBufferRef(targets),
		scratchBuffer,
		outBuffer,
		format,
		kernel,
		count,
	); err != nil {
		host.dispatchError(err)
	}

	return host.bridge.readFloat32Scalar(outBuffer)
}

func (host *ComputeHost) CrossEntropyScalar(
	logits, targets unsafe.Pointer,
	batchSize, classes int,
	format dtype.DType,
) float32 {
	if batchSize == 0 || classes == 0 || host.bridge == nil {
		return 0
	}

	batch := uint32(batchSize)
	scratchBuffer := host.bridge.borrowScratch(crossEntropyScratchBytes(batch))
	outBuffer := host.bridge.borrowScratch(4)

	defer host.bridge.releaseScratch(scratchBuffer)
	defer host.bridge.releaseScratch(outBuffer)

	if err := losses.DispatchCrossEntropy(
		host.contextRef(),
		resolveBufferRef(logits),
		resolveBufferRef(targets),
		scratchBuffer,
		outBuffer,
		nil,
		format,
		batch,
		uint32(classes),
	); err != nil {
		host.dispatchError(err)
	}

	return host.bridge.readFloat32Scalar(outBuffer)
}

func samplingRandomTarget(seed uint64) float32 {
	source := rand.NewChaCha8([32]byte{
		byte(seed), byte(seed >> 8), byte(seed >> 16), byte(seed >> 24),
		byte(seed >> 32), byte(seed >> 40), byte(seed >> 48), byte(seed >> 56),
	})

	return rand.New(source).Float32()
}

func effectiveSamplingCount(kernel sampling.SamplingKernel, config device.SamplingConfig, count uint32) uint32 {
	if kernel != sampling.KernelTopK {
		return count
	}

	topK := config.TopK

	if topK <= 0 || topK > int(count) {
		return count
	}

	return uint32(topK)
}

func (host *ComputeHost) SamplingIndex(
	kernel sampling.SamplingKernel,
	config device.SamplingConfig,
	logits unsafe.Pointer,
	vocabSize int,
	format dtype.DType,
) int32 {
	if vocabSize == 0 || host.bridge == nil {
		return 0
	}

	count := uint32(vocabSize)
	outBuffer := host.bridge.borrowScratch(4)

	defer host.bridge.releaseScratch(outBuffer)

	if kernel == sampling.KernelGreedy {
		if err := sampling.DispatchSampling(
			host.contextRef(),
			0,
			resolveBufferRef(logits),
			nil,
			nil,
			outBuffer,
			format,
			count,
			0,
		); err != nil {
			host.dispatchError(err)
		}

		return host.bridge.readInt32Scalar(outBuffer)
	}

	effectiveCount := effectiveSamplingCount(kernel, config, count)
	padded := sampling.PaddedCount(effectiveCount)
	scoresBuffer := host.bridge.borrowScratch(samplingScoresScratchBytes(padded))
	indicesBuffer := host.bridge.borrowScratch(samplingIndicesScratchBytes(padded))

	defer host.bridge.releaseScratch(scoresBuffer)
	defer host.bridge.releaseScratch(indicesBuffer)

	target := samplingRandomTarget(config.Seed)

	if err := sampling.DispatchSampling(
		host.contextRef(),
		1,
		resolveBufferRef(logits),
		scoresBuffer,
		indicesBuffer,
		outBuffer,
		format,
		effectiveCount,
		target,
	); err != nil {
		host.dispatchError(err)
	}

	return host.bridge.readInt32Scalar(outBuffer)
}

func (host *ComputeHost) DispatchSimilarity(
	left, right unsafe.Pointer,
	count int,
	format dtype.DType,
) float32 {
	return host.DotProduct(left, right, count, format)
}
