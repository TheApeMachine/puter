//go:build cuda

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
