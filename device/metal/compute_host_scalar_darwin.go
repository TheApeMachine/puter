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

	if err := reduction.DispatchReductionRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(values))),
		uintptr(unsafe.Pointer(scratchA)),
		uintptr(unsafe.Pointer(scratchB)),
		uintptr(unsafe.Pointer(resolveBufferRef(dst))),
		format,
		kernel,
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

	if err := losses.DispatchPairLossRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(predictions))),
		uintptr(unsafe.Pointer(resolveBufferRef(targets))),
		uintptr(unsafe.Pointer(scratchBuffer)),
		uintptr(unsafe.Pointer(resolveBufferRef(dst))),
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

	if err := losses.DispatchCrossEntropyRefs(
		host.contextRef(),
		uintptr(unsafe.Pointer(resolveBufferRef(logits))),
		uintptr(unsafe.Pointer(resolveBufferRef(targets))),
		uintptr(unsafe.Pointer(scratchBuffer)),
		uintptr(unsafe.Pointer(resolveBufferRef(dst))),
		format,
		batch,
		uint32(classes),
	); err != nil {
		host.dispatchError(err)
	}
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

		return
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
}

func (host *ComputeHost) DotProduct(
	dst unsafe.Pointer,
	left, right unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	host.unavailable()
}

func (host *ComputeHost) DispatchSimilarity(
	dst unsafe.Pointer,
	left, right unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	host.unavailable()
}
