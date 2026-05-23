//go:build !cuda

package dequant

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
)

func (dequantization *Dequantization) Dequant(
	dst, src unsafe.Pointer,
	count int,
	config device.DequantInt8Config,
	dstFormat, srcFormat dtype.DType,
) {
	dequantization.stubHost()
}
