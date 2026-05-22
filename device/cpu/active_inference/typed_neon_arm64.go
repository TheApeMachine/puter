//go:build arm64

package active_inference

import "github.com/theapemachine/manifesto/dtype"

//go:noescape
func PrecisionWeightBFloat16NEONAsm(errors, precision, output *dtype.BF16, count int)

//go:noescape
func BeliefUpdateBFloat16NEONAsm(likelihood, prior, output *dtype.BF16, count int)

//go:noescape
func FreeEnergyBFloat16NEONAsm(likelihood, posterior, prior *dtype.BF16, count int) dtype.BF16

//go:noescape
func ExpectedFreeEnergyBFloat16NEONAsm(
	predictedObs, preferredObs, predictedState *dtype.BF16,
	obsCount, stateCount int,
) dtype.BF16

//go:noescape
func PrecisionWeightFloat16NEONAsm(errors, precision, output *dtype.F16, count int)

//go:noescape
func BeliefUpdateFloat16NEONAsm(likelihood, prior, output *dtype.F16, count int)

//go:noescape
func FreeEnergyFloat16NEONAsm(likelihood, posterior, prior *dtype.F16, count int) dtype.F16

//go:noescape
func ExpectedFreeEnergyFloat16NEONAsm(
	predictedObs, preferredObs, predictedState *dtype.F16,
	obsCount, stateCount int,
) dtype.F16
