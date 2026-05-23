//go:build cuda

package sampling

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

func (sampling *Sampling) GreedySample(
	logits unsafe.Pointer,
	vocabSize int,
	format dtype.DType,
) int32 {
	return sampling.host.SamplingIndex(
		KernelGreedy,
		device.SamplingConfig{},
		logits,
		vocabSize,
		format,
	)
}
