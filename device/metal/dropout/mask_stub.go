//go:build !darwin || !cgo

package dropout

import (
	"unsafe"
	
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

func (dropoutLayer *DropoutLayer) Dropout(dst, src unsafe.Pointer, count int, config device.DropoutConfig, format dtype.DType,) {
	dropoutLayer.stubHost()
}

