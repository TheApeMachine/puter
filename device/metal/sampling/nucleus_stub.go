//go:build !darwin || !cgo

package sampling

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

func (sampling *Sampling) TopKSample(config device.SamplingConfig, logits unsafe.Pointer, vocabSize int, format dtype.DType) int32 {
	sampling.stubHost()
	return 0
}

func (sampling *Sampling) TopPSample(config device.SamplingConfig, logits unsafe.Pointer, vocabSize int, format dtype.DType) int32 {
	sampling.stubHost()
	return 0
}
