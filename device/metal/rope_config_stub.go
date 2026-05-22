//go:build !darwin || !cgo

package metal

import (
	"errors"

	"github.com/theapemachine/manifesto/tensor"
)

func runMetalRoPEConfigured(
	input tensor.Tensor,
	out tensor.Tensor,
	config RoPEConfig,
) error {
	_ = input
	_ = out
	_ = config

	return errors.New("metal rope requires darwin with cgo")
}
