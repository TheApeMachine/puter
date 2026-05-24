//go:build darwin && cgo

package metal

import (
	"crypto/sha256"
	"encoding/binary"
	"math/rand/v2"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
	"github.com/theapemachine/puter/device/metal/losses"
	"github.com/theapemachine/puter/device/metal/reduction"
	"github.com/theapemachine/puter/device/metal/sampling"
)

func samplingChaCha8Key(seed uint64) [32]byte {
	var input [8]byte
	binary.LittleEndian.PutUint64(input[:], seed)

	return sha256.Sum256(input[:])
}

func samplingRandomTarget(seed uint64) float32 {
	source := rand.NewChaCha8(samplingChaCha8Key(seed))

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

	if err := reduction.DispatchReductionRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(values))),
		uintptr(unsafe.Pointer(scratchA)),
		uintptr(unsafe.Pointer(scratchB)),
		uintptr(unsafe.Pointer(outBuffer)),
		format,
		kernel,
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

	if err := losses.DispatchPairLossRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(predictions))),
		uintptr(unsafe.Pointer(resolveBufferRef(targets))),
		uintptr(unsafe.Pointer(scratchBuffer)),
		uintptr(unsafe.Pointer(outBuffer)),
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

	if err := losses.DispatchCrossEntropyRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(logits))),
		uintptr(unsafe.Pointer(resolveBufferRef(targets))),
		uintptr(unsafe.Pointer(scratchBuffer)),
		uintptr(unsafe.Pointer(outBuffer)),
		format,
		batch,
		uint32(classes),
	); err != nil {
		host.dispatchError(err)
	}

	return host.bridge.readFloat32Scalar(outBuffer)
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
		if err := sampling.DispatchSamplingRefs(
			host.contextRef(),
			sampling.OperationGreedy,
			uintptr(unsafe.Pointer(resolveBufferRef(logits))),
			0,
			0,
			uintptr(unsafe.Pointer(outBuffer)),
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

	if err := sampling.DispatchSamplingRefs(
		host.contextRef(),
		sampling.OperationProbabilistic,
		uintptr(unsafe.Pointer(resolveBufferRef(logits))),
		uintptr(unsafe.Pointer(scoresBuffer)),
		uintptr(unsafe.Pointer(indicesBuffer)),
		uintptr(unsafe.Pointer(outBuffer)),
		format,
		effectiveCount,
		target,
	); err != nil {
		host.dispatchError(err)
	}

	return host.bridge.readInt32Scalar(outBuffer)
}

func (host *ComputeHost) DotProduct(
	left, right unsafe.Pointer,
	count int,
	format dtype.DType,
) float32 {
	host.unavailable()
	return 0
}

func (host *ComputeHost) DispatchSimilarity(
	left, right unsafe.Pointer,
	count int,
	format dtype.DType,
) float32 {
	host.unavailable()
	return 0
}
