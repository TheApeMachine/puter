//go:build xla

package sampling

import (
	"unsafe"
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

func (sampling *Sampling) TopKSample(config device.SamplingConfig, logits unsafe.Pointer, vocabSize int, format dtype.DType) int32 {
	sampling.unimplemented("TopKSample")
	return 0
}

func (sampling *Sampling) TopPSample(config device.SamplingConfig, logits unsafe.Pointer, vocabSize int, format dtype.DType) int32 {
	sampling.unimplemented("TopPSample")
	return 0
}

func (sampling *Sampling) GreedySample(logits unsafe.Pointer, vocabSize int, format dtype.DType) int32 {
	sampling.unimplemented("GreedySample")
	return 0
}

