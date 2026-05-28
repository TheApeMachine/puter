//go:build !darwin || !cgo

package sampling

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (sampling *Sampling) GreedySample(dst, logits unsafe.Pointer, vocabSize int, format dtype.DType) {
	sampling.stubHost()
}
