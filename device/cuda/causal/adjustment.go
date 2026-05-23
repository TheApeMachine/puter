//go:build cuda

package causal

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
)

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
	causal.host.DispatchFrontdoorAdjustment(mediatorGivenX, outcomeGivenXM, marginalX, output, xCount, mediatorCount, yCount, format)
}
