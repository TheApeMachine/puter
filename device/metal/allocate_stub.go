//go:build !darwin || !cgo

package metal

import (
	"unsafe"

	"github.com/theapemachine/manifesto/tensor"
)

func metalTensorContents(value tensor.Tensor) unsafe.Pointer {
	_ = value

	return nil
}

func metalMemset(destination unsafe.Pointer, value byte, byteCount int) {
	_ = destination
	_ = value
	_ = byteCount
}
