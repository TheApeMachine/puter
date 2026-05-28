//go:build !xla

package dot

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (product *Product) Dot(dst, left, right unsafe.Pointer, count int, format dtype.DType) {
	product.stubHost()
}
