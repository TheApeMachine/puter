//go:build !cuda

package active_inference

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (activeInference *ActiveInference) BeliefUpdate(
	likelihood, prior, output unsafe.Pointer,
	count int,
	format dtype.DType,
) {
	activeInference.stubHost()
}
