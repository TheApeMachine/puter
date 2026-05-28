//go:build xla

package sampling

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

/*
Sampling methods write the sampled token index into `*dst` as int32
(ARCHITECTURE.md §2.2). The XLA host currently materializes the result
through PJRT to a host int32 and stores it at dst; once the planner /
executable cache work lands (GAPS.md §2.5–2.6) the result will be
written directly into the caller's PjRtBuffer slot.
*/
func (sampling *Sampling) TopKSample(
	dst, logits unsafe.Pointer,
	vocabSize int,
	config device.SamplingConfig,
	format dtype.DType,
) {
	sampling.host.DispatchTopKSample(dst, config, logits, vocabSize, format)
}

func (sampling *Sampling) TopPSample(
	dst, logits unsafe.Pointer,
	vocabSize int,
	config device.SamplingConfig,
	format dtype.DType,
) {
	sampling.host.DispatchTopPSample(dst, config, logits, vocabSize, format)
}

func (sampling *Sampling) GreedySample(
	dst, logits unsafe.Pointer,
	vocabSize int,
	format dtype.DType,
) {
	sampling.host.DispatchGreedySample(dst, logits, vocabSize, format)
}
