//go:build !darwin || !cgo

package quant

import (
	"unsafe"
	
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

func (quantization *Quantization) Quant(dst, src unsafe.Pointer, count int, config device.DequantInt8Config, dstFormat, srcFormat dtype.DType) {
	quantization.stubHost()
}

