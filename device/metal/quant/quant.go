package quant

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

/*
Quantization implements device.Quant for the Metal backend.
*/
type Quantization struct {
	host Host
}

func New(host Host) Quantization {
	return Quantization{host: host}
}

type Host interface {
	NeedsPlatform()
	DispatchQuant(dst, src unsafe.Pointer, count int, config device.DequantInt8Config, dstFormat, srcFormat dtype.DType)
}
