//go:build !darwin || !cgo

package causal

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (causal *Causal) DoIntervene(adjacency, intervened, output unsafe.Pointer, nodeCount, intervenedCount int, format dtype.DType,) {
	causal.stubHost()
}

func (causal *Causal) CATE(treated, control, output unsafe.Pointer, count int, format dtype.DType) {
	causal.stubHost()
}

func (causal *Causal) Counterfactual(observedY, observedX, counterfactualX, output unsafe.Pointer, count int, slope float32, format dtype.DType,) {
	causal.stubHost()
}

func (causal *Causal) IVEstimate(instrument, treatment, outcome unsafe.Pointer, count int, output unsafe.Pointer, format dtype.DType,) {
	causal.stubHost()
}
