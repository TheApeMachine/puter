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
	dst unsafe.Pointer,
	values unsafe.Pointer,
	count int,
	format dtype.DType,
	kernel reduction.ReductionKernel,
) {
	if count == 0 || host.bridge == nil {
		return
	}

	elementCount := uint32(count)
	scratchA := host.bridge.borrowScratch(reductionScratchBytes(elementCount))
	scratchB := host.bridge.borrowScratch(reductionScratchBytes(elementCount))

	defer host.bridge.releaseScratch(scratchA)
	defer host.bridge.releaseScratch(scratchB)

	if err := reduction.DispatchReduction(
		host.contextRef(),
		resolveBufferRef(values),
		scratchA,
		scratchB,
		resolveBufferRef(dst),
		format,
		kernel,
		elementCount,
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) DotProduct(
	dst unsafe.Pointer,
	left, right unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	if count == 0 || host.bridge == nil {
		return
	}

	elementCount := uint32(count)
	scratchBuffer := host.bridge.borrowScratch(dotScratchBytes(elementCount))

	defer host.bridge.releaseScratch(scratchBuffer)

	if err := dot.DispatchDot(
		host.contextRef(),
		resolveBufferRef(left),
		resolveBufferRef(right),
		scratchBuffer,
		resolveBufferRef(dst),
		format,
		elementCount,
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) PairLossScalar(
	dst unsafe.Pointer,
	predictions, targets unsafe.Pointer,
	count int,
	format dtype.DType,
	kernel losses.LossKernel,
) {
	if count == 0 || host.bridge == nil {
		return
	}

	elementCount := uint32(count)
	scratchBuffer := host.bridge.borrowScratch(reductionScratchBytes(elementCount))

	defer host.bridge.releaseScratch(scratchBuffer)

	if err := losses.DispatchPairLoss(
		host.contextRef(),
		resolveBufferRef(predictions),
		resolveBufferRef(targets),
		scratchBuffer,
		resolveBufferRef(dst),
		format,
		kernel,
		elementCount,
	); err != nil {
		host.dispatchError(err)
	}
}

func (host *ComputeHost) CrossEntropyScalar(
	dst unsafe.Pointer,
	logits, targets unsafe.Pointer,
	batchSize, classes int,
	format dtype.DType,
) {
	if batchSize == 0 || classes == 0 || host.bridge == nil {
		return
	}

	batch := uint32(batchSize)
	scratchBuffer := host.bridge.borrowScratch(crossEntropyScratchBytes(batch))

	defer host.bridge.releaseScratch(scratchBuffer)

	if err := losses.DispatchCrossEntropy(
		host.contextRef(),
		resolveBufferRef(logits),
		resolveBufferRef(targets),
		scratchBuffer,
		resolveBufferRef(dst),
		nil,
		format,
		batch,
		uint32(classes),
	); err != nil {
		host.dispatchError(err)
	}
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
	dst unsafe.Pointer,
	kernel sampling.SamplingKernel,
	config device.SamplingConfig,
	logits unsafe.Pointer,
	vocabSize int,
	format dtype.DType,
) {
	if vocabSize == 0 || host.bridge == nil {
		return
	}

	count := uint32(vocabSize)
	outBuffer := resolveBufferRef(dst)

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

		return
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
}

func (host *ComputeHost) DispatchSimilarity(
	dst unsafe.Pointer,
	left, right unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	host.DotProduct(dst, left, right, count, format)
}
