//go:build !xla

package causal

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)
func (causal *Causal) Cholesky(input, output unsafe.Pointer, matrixOrder int, format dtype.DType) {
	causal.stubHost()
}

func (causal *Causal) BackdoorAdjustment( conditional, marginalZ, output unsafe.Pointer, xCount, zCount, yCount int, format dtype.DType, ) {
	causal.stubHost()
}

func (causal *Causal) FrontdoorAdjustment( mediatorGivenX, outcomeGivenXM, marginalX, output unsafe.Pointer, xCount, mediatorCount, yCount int, format dtype.DType, ) {
	causal.stubHost()
}

func (causal *Causal) DoIntervene( adjacency, intervened, output unsafe.Pointer, nodeCount, intervenedCount int, format dtype.DType, ) {
	causal.stubHost()
}

func (causal *Causal) CATE(treated, control, output unsafe.Pointer, count int, format dtype.DType) {
	causal.stubHost()
}

func (causal *Causal) Counterfactual( observedY, observedX, counterfactualX, output unsafe.Pointer, count int, slope float32, format dtype.DType, ) {
	causal.stubHost()
}

func (causal *Causal) IVEstimate( instrument, treatment, outcome unsafe.Pointer, count int, output unsafe.Pointer, format dtype.DType, ) {
	causal.stubHost()
}

func (causal *Causal) DAGMarkovFactorization( conditionals unsafe.Pointer, conditionalCount int, output unsafe.Pointer, format dtype.DType, ) {
	causal.stubHost()
}

func (causal *Causal) MarkovFlowActive( mutualInformation, partition, output unsafe.Pointer, nodeCount int, format dtype.DType, ) {
	causal.stubHost()
}

func (causal *Causal) MarkovFlowInternal( mutualInformation, partition, output unsafe.Pointer, nodeCount int, format dtype.DType, ) {
	causal.stubHost()
}

