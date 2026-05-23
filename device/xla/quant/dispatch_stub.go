//go:build !xla

package quant

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)
func (quantization *Quantization) Quant(dst, src unsafe.Pointer, count int, config DequantInt8Config, dstFormat, srcFormat dtype.DType) {
	quantization.stubHost()
}

