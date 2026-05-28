package sampling

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cpu/internal/scalar"
)

func requireSamplingFloat32(format dtype.DType) {
	if format != dtype.Float32 {
		panic("sampling: unsupported dtype")
	}
}

/*
GreedySample writes the argmax index of `logits` into `*dst` as int32.
Zero-host-sync per ARCHITECTURE.md §2.2.
*/
func (sampling Sampling) GreedySample(dst, logits unsafe.Pointer, vocabSize int, format dtype.DType) {
	if vocabSize == 0 {
		scalar.StoreInt32(dst, 0)
		return
	}

	requireSamplingFloat32(format)

	logitView := unsafe.Slice((*float32)(logits), vocabSize)
	scalar.StoreInt32(dst, GreedySampleFloat32Native(logitView))
}

/*
TopKSample writes the sampled token index into `*dst` as int32.
*/
func (sampling Sampling) TopKSample(dst, logits unsafe.Pointer, vocabSize int, config SamplingConfig, format dtype.DType) {
	if vocabSize == 0 {
		scalar.StoreInt32(dst, 0)
		return
	}

	requireSamplingFloat32(format)

	logitView := unsafe.Slice((*float32)(logits), vocabSize)
	topK := config.TopK

	if topK <= 0 || topK > vocabSize {
		topK = vocabSize
	}

	scalar.StoreInt32(dst, TopKSampleFloat32Native(logitView, config.Temperature, topK, config.Seed))
}

/*
TopPSample writes the sampled token index into `*dst` as int32.
*/
func (sampling Sampling) TopPSample(dst, logits unsafe.Pointer, vocabSize int, config SamplingConfig, format dtype.DType) {
	if vocabSize == 0 {
		scalar.StoreInt32(dst, 0)
		return
	}

	requireSamplingFloat32(format)

	logitView := unsafe.Slice((*float32)(logits), vocabSize)
	scalar.StoreInt32(dst, TopPSampleFloat32Native(logitView, config.Temperature, config.TopP, config.Seed))
}
