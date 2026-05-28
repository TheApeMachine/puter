//go:build !cuda

package sampling

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

func (sampling *Sampling) TopKSample(dst, logits unsafe.Pointer, vocabSize int, config device.SamplingConfig, format dtype.DType) {
	sampling.stubHost()
}

func (sampling *Sampling) TopPSample(dst, logits unsafe.Pointer, vocabSize int, config device.SamplingConfig, format dtype.DType) {
	sampling.stubHost()
}
