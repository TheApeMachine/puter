//go:build cuda

package sampling

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

func (sampling *Sampling) TopKSample(
	config device.SamplingConfig,
	logits unsafe.Pointer,
	vocabSize int,
	format dtype.DType,
) int32 {
	return sampling.host.SamplingIndex(KernelTopK, config, logits, vocabSize, format)
}

func (sampling *Sampling) TopPSample(
	config device.SamplingConfig,
	logits unsafe.Pointer,
	vocabSize int,
	format dtype.DType,
) int32 {
	return sampling.host.SamplingIndex(KernelTopP, config, logits, vocabSize, format)
}
