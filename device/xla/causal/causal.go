package causal

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

/*
Causal implements device.Causal for the XLA backend.
*/
type Causal struct {
	host Host
}

/*
Host is the XLA dispatch surface causal operations call into.
*/
type Host interface {
	NeedsPlatform()
	DispatchBackdoorAdjustment(
		conditional, marginalZ, output unsafe.Pointer,
		xCount, zCount, yCount int,
		format dtype.DType,
	)
	DispatchFrontdoorAdjustment(
		mediatorGivenX, outcomeGivenXM, marginalX, output unsafe.Pointer,
		xCount, mediatorCount, yCount int,
		format dtype.DType,
	)
	DispatchCholesky(input, output unsafe.Pointer, matrixOrder int, format dtype.DType)
	DispatchDoIntervene(
		adjacency, intervened, output unsafe.Pointer,
		nodeCount, intervenedCount int,
		format dtype.DType,
	)
	DispatchCATE(treated, control, output unsafe.Pointer, count int, format dtype.DType)
	DispatchCounterfactual(
		observedY, observedX, counterfactualX, output unsafe.Pointer,
		count int,
		slope float32,
		format dtype.DType,
	)
	DispatchIVEstimate(
		instrument, treatment, outcome, output unsafe.Pointer,
		count int,
		format dtype.DType,
	)
	DispatchDAGMarkovFactorization(
		conditionals unsafe.Pointer,
		conditionalCount int,
		output unsafe.Pointer,
		format dtype.DType,
	)
	DispatchMarkovFlowActive(
		mutualInformation, partition, output unsafe.Pointer,
		nodeCount int,
		format dtype.DType,
	)
	DispatchMarkovFlowInternal(
		mutualInformation, partition, output unsafe.Pointer,
		nodeCount int,
		format dtype.DType,
	)
	NotImplemented(string)
}

/*
New wires a Causal receiver to its XLA dispatch host.
*/
func New(host Host) Causal {
	return Causal{host: host}
}

func (receiver *Causal) stubHost() {
	receiver.host.NeedsPlatform()
}

func (receiver *Causal) unimplemented(methodName string) {
	receiver.host.NotImplemented(methodName)
}
