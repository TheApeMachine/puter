//go:build !darwin || !cgo

package dot

import (
	"unsafe"
	
	"github.com/theapemachine/manifesto/dtype"
)

func (product *Product) Dot(left, right unsafe.Pointer, count int, format dtype.DType,) float32 {
	product.stubHost()
	return 0
}

