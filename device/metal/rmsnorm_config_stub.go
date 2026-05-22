//go:build !darwin || !cgo

package metal

import (
	"errors"

	"github.com/theapemachine/manifesto/tensor"
)

func runMetalRMSNormConfigured(
	input tensor.Tensor,
	scale tensor.Tensor,
	out tensor.Tensor,
	config RMSNormConfig,
) error {
	_ = input
	_ = scale
	_ = out
	_ = config

	return errors.New("metal rmsnorm requires darwin with cgo")
}
