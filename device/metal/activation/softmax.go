//go:build darwin && cgo

package activation

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (activation *Activation) Softmax(dst, src unsafe.Pointer, count int, format dtype.DType) {
	_ = count
	activation.host.Softmax(dst, src, format)
}

func (activation *Activation) LogSoftmax(dst, src unsafe.Pointer, count int, format dtype.DType) {
	_ = count
	activation.host.Softmax(dst, src, format)
	activation.host.StandardUnary(dst, dst, format, StandardLog)
}
