//go:build xla

package elementwise

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)
func (elementwise *Elementwise) Axpy(y, x unsafe.Pointer, count int, alpha float32, format dtype.DType) {
	elementwise.unimplemented("Axpy")
}

