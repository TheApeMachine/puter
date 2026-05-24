package causal

import (
	"unsafe"
	"github.com/theapemachine/manifesto/dtype"
)

var defaultCausal = New()

func BackdoorAdjustment(conditional, marginalZ, output unsafe.Pointer,
	xCount, zCount, yCount int,
	format dtype.DType) {
	defaultCausal.BackdoorAdjustment(conditional, marginalZ, output, xCount, zCount, yCount, format)
}

func CATE(treated, control, output unsafe.Pointer, count int, format dtype.DType) {
	defaultCausal.CATE(treated, control, output, count, format)
}

func Cholesky(input, output unsafe.Pointer, matrixOrder int, format dtype.DType) {
	defaultCausal.Cholesky(input, output, matrixOrder, format)
}

func Counterfactual(observedY, observedX, counterfactualX, output unsafe.Pointer,
	count int,
	slope float32,
	format dtype.DType) {
	defaultCausal.Counterfactual(observedY, observedX, counterfactualX, output, count, slope, format)
}

func DAGMarkovFactorization(conditionals unsafe.Pointer,
	conditionalCount int,
	output unsafe.Pointer,
	format dtype.DType) {
	defaultCausal.DAGMarkovFactorization(conditionals, conditionalCount, output, format)
}

func DoIntervene(adjacency, intervened, output unsafe.Pointer,
	nodeCount, intervenedCount int,
	format dtype.DType) {
	defaultCausal.DoIntervene(adjacency, intervened, output, nodeCount, intervenedCount, format)
}

func FrontdoorAdjustment(mediatorGivenX, outcomeGivenXM, marginalX, output unsafe.Pointer,
	xCount, mediatorCount, yCount int,
	format dtype.DType) {
	defaultCausal.FrontdoorAdjustment(mediatorGivenX, outcomeGivenXM, marginalX, output, xCount, mediatorCount, yCount, format)
}

func IVEstimate(instrument, treatment, outcome unsafe.Pointer,
	count int,
	output unsafe.Pointer,
	format dtype.DType) {
	defaultCausal.IVEstimate(instrument, treatment, outcome, count, output, format)
}

func MarkovFlowActive(mutualInformation, partition, output unsafe.Pointer,
	nodeCount int,
	format dtype.DType) {
	defaultCausal.MarkovFlowActive(mutualInformation, partition, output, nodeCount, format)
}

func MarkovFlowInternal(mutualInformation, partition, output unsafe.Pointer,
	nodeCount int,
	format dtype.DType) {
	defaultCausal.MarkovFlowInternal(mutualInformation, partition, output, nodeCount, format)
}
