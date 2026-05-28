//go:build darwin && cgo

package sampling

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

func (sampling *Sampling) GreedySample(
	dst, logits unsafe.Pointer,
	vocabSize int,
	format dtype.DType,
) {
	sampling.host.SamplingIndex(
		dst,
		KernelGreedy,
		device.SamplingConfig{},
		logits,
		vocabSize,
		format,
	)
}
