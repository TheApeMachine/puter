//go:build xla

package causal

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

func (causalModel *Causal) MarkovFlowActive(
	mutualInformation, partition, output unsafe.Pointer,
	nodeCount int,
	format dtype.DType,
) {
	causalModel.host.DispatchMarkovFlowActive(mutualInformation, partition, output, nodeCount, format)
}

func (causalModel *Causal) MarkovFlowInternal(
	mutualInformation, partition, output unsafe.Pointer,
	nodeCount int,
	format dtype.DType,
) {
	causalModel.host.DispatchMarkovFlowInternal(mutualInformation, partition, output, nodeCount, format)
}

func (causal *Causal) BackdoorAdjustment(
	conditional, marginalZ, output unsafe.Pointer,
	xCount, zCount, yCount int,
	format dtype.DType,
) {
	causal.host.DispatchBackdoorAdjustment(conditional, marginalZ, output, xCount, zCount, yCount, format)
}

func (causal *Causal) FrontdoorAdjustment(
	mediatorGivenX, outcomeGivenXM, marginalX, output unsafe.Pointer,
	xCount, mediatorCount, yCount int,
	format dtype.DType,
) {
	causal.host.DispatchFrontdoorAdjustment(
		mediatorGivenX, outcomeGivenXM, marginalX, output,
		xCount, mediatorCount, yCount, format,
	)
}

func (causal *Causal) Cholesky(input, output unsafe.Pointer, matrixOrder int, format dtype.DType) {
	causal.host.DispatchCholesky(input, output, matrixOrder, format)
}

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
	causal.host.DispatchIVEstimate(instrument, treatment, outcome, output, count, format)
}

func (causal *Causal) DAGMarkovFactorization(
	conditionals unsafe.Pointer,
	conditionalCount int,
	output unsafe.Pointer,
	format dtype.DType,
) {
	causal.host.DispatchDAGMarkovFactorization(conditionals, conditionalCount, output, format)
}
