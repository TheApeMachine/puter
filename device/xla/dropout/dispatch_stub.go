//go:build !xla

package dropout

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)
func (dropoutLayer *DropoutLayer) Dropout( dst, src unsafe.Pointer, count int, config DropoutConfig, format dtype.DType, ) {
	dropoutLayer.stubHost()
}

