//go:build cuda

package causal

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (causal *Causal) Cholesky(input, output unsafe.Pointer, matrixOrder int, format dtype.DType) {
	causal.host.DispatchCholesky(input, output, matrixOrder, format)
}
