//go:build arm64

package active_inference

//go:noescape
func PrecisionWeightBFloat16NEONAsm(errors, precision, output *uint16, count int)

//go:noescape
func BeliefUpdateBFloat16NEONAsm(likelihood, prior, output *uint16, count int)

//go:noescape
func freeEnergyBFloat16NEONBridge(likelihood, posterior, prior uintptr, count int) uint16

//go:noescape
func expectedFreeEnergyBFloat16NEONBridge(
	predictedObs, preferredObs, predictedState uintptr,
	obsCount, stateCount int,
) uint16

//go:noescape
func PrecisionWeightFloat16NEONAsm(errors, precision, output *uint16, count int)

//go:noescape
func BeliefUpdateFloat16NEONAsm(likelihood, prior, output *uint16, count int)

//go:noescape
func freeEnergyFloat16NEONBridge(likelihood, posterior, prior uintptr, count int) uint16

//go:noescape
func expectedFreeEnergyFloat16NEONBridge(
	predictedObs, preferredObs, predictedState uintptr,
	obsCount, stateCount int,
) uint16
