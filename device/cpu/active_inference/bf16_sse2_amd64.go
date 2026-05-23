//go:build amd64

package active_inference

import "github.com/theapemachine/manifesto/dtype"

//go:noescape
func PrecisionWeightBFloat16SSE2Asm(errors, precision, output *uint16, count int)

//go:noescape
func BeliefUpdateBFloat16SSE2Asm(likelihood, prior, output *uint16, count int)

func PrecisionWeightBF16SSE2(errors, precision, output []dtype.BF16) {
	if len(errors) == 0 {
		return
	}

	PrecisionWeightBFloat16SSE2Asm(
		(*uint16)(&errors[0]),
		(*uint16)(&precision[0]),
		(*uint16)(&output[0]),
		len(errors),
	)
}

func BeliefUpdateBF16SSE2(likelihood, prior, output []dtype.BF16) {
	if len(likelihood) == 0 {
		return
	}

	BeliefUpdateBFloat16SSE2Asm(
		(*uint16)(&likelihood[0]),
		(*uint16)(&prior[0]),
		(*uint16)(&output[0]),
		len(likelihood),
	)
}
