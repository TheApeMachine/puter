//go:build !cuda

package matmul

import (
	"unsafe"
	
	"github.com/theapemachine/manifesto/dtype"
)

func (gemm *Gemm) Matmul(out, left, right unsafe.Pointer, rows, inner, cols int, format dtype.DType,) {
	gemm.stubHost()
}

