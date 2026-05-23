//go:build !cuda

package activation

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (activation *Activation) Softmax(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}

func (activation *Activation) LogSoftmax(dst, src unsafe.Pointer, count int, format dtype.DType) {
	activation.stubHost()
}
