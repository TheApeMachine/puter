//go:build darwin && cgo

package sampling

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

/*
GreedySample writes the argmax token index into `*dst` as int32
(ARCHITECTURE.md §2.2). The Metal host currently computes on device
and reads back the int32; once the static memory planner lands
(GAPS.md P1) the host signature will take the workspace-resolved
MetalBufferRef directly, eliminating the device→host round-trip.
*/
func (sampling *Sampling) GreedySample(
	dst, logits unsafe.Pointer,
	vocabSize int,
	format dtype.DType,
) {
	*(*int32)(dst) = sampling.host.SamplingIndex(
		KernelGreedy,
		device.SamplingConfig{},
		logits,
		vocabSize,
		format,
	)
}
