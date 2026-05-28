//go:build darwin && cgo

package sampling

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

func (sampling *Sampling) TopKSample(
	dst, logits unsafe.Pointer,
	vocabSize int,
	config device.SamplingConfig,
	format dtype.DType,
) {
	sampling.host.SamplingIndex(dst, KernelTopK, config, logits, vocabSize, format)
}

func (sampling *Sampling) TopPSample(
	dst, logits unsafe.Pointer,
	vocabSize int,
	config device.SamplingConfig,
	format dtype.DType,
) {
	sampling.host.SamplingIndex(dst, KernelTopP, config, logits, vocabSize, format)
}
