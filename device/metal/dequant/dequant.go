package dequant

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

/*
Dequantization implements device.Dequant for the Metal backend.
*/
type Dequantization struct {
	host Host
}

func New(host Host) Dequantization {
	return Dequantization{host: host}
}

type Host interface {
	NeedsPlatform()
	DispatchDequant(dst, src unsafe.Pointer, count int, config device.DequantInt8Config, dstFormat, srcFormat dtype.DType)
	DispatchDequant4(dst, src unsafe.Pointer, pairCount int, config device.DequantInt4Config, dstFormat, srcFormat dtype.DType)
}
