//go:build cuda

package causal

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (causal *Causal) DAGMarkovFactorization(
	conditionals unsafe.Pointer,
	conditionalCount int,
	output unsafe.Pointer,
	format dtype.DType,
) {
	causal.host.DispatchDAGMarkovFactorization(conditionals, conditionalCount, output, format)
}
