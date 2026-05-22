//go:build !darwin || !cgo

package metal

import (
	"github.com/theapemachine/manifesto/tensor"
)

func runMetalWeightGraftAdd(weights tensor.Tensor, injection tensor.Tensor) error {
	_ = weights
	_ = injection

	return tensor.ErrNeedsPlatformSetup
}
