//go:build cuda

package causal

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (causal *Causal) DoIntervene(
	adjacency, intervened, output unsafe.Pointer,
	nodeCount, intervenedCount int,
	format dtype.DType,
) {
	causal.host.DispatchDoIntervene(adjacency, intervened, output, nodeCount, intervenedCount, format)
}

func (causal *Causal) CATE(treated, control, output unsafe.Pointer, count int, format dtype.DType) {
	causal.host.DispatchCATE(treated, control, output, count, format)
}

func (causal *Causal) Counterfactual(
	observedY, observedX, counterfactualX, output unsafe.Pointer,
	count int,
	slope float32,
	format dtype.DType,
) {
	causal.host.DispatchCounterfactual(observedY, observedX, counterfactualX, output, count, slope, format)
}

func (causal *Causal) IVEstimate(
	instrument, treatment, outcome unsafe.Pointer,
	count int,
	output unsafe.Pointer,
	format dtype.DType,
) {
	causal.host.DispatchIVEstimate(instrument, treatment, outcome, count, output, format)
}
