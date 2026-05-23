//go:build !darwin || !cgo

package sampling

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (sampling *Sampling) GreedySample(logits unsafe.Pointer, vocabSize int, format dtype.DType) int32 {
	sampling.stubHost()
	return 0
}
