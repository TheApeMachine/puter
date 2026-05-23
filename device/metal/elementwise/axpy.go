//go:build darwin && cgo

package elementwise

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (elementwise *Elementwise) Axpy(
	y, x unsafe.Pointer,
	count int,
	alpha float32,
	format dtype.DType,
) {
	_ = count
	elementwise.host.DispatchAxpy(y, x, alpha, format)
}
