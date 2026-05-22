//go:build !darwin || !cgo

package metal

import (
	"github.com/theapemachine/manifesto/tensor"
)

func runMetalActivationSteer(
	base tensor.Tensor,
	direction tensor.Tensor,
	coefficient tensor.Tensor,
	out tensor.Tensor,
) error {
	_ = base
	_ = direction
	_ = coefficient
	_ = out

	return tensor.ErrNeedsPlatformSetup
}
