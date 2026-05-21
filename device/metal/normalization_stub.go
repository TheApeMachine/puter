//go:build !darwin || !cgo

package metal

import "github.com/theapemachine/manifesto/tensor"

func runMetalLayerNorm(
	tensor.Tensor,
	tensor.Tensor,
	tensor.Tensor,
	tensor.Tensor,
) error {
	return tensor.ErrNeedsPlatformSetup
}

func runMetalRMSNorm(tensor.Tensor, tensor.Tensor, tensor.Tensor) error {
	return tensor.ErrNeedsPlatformSetup
}

func runMetalGroupNorm(
	tensor.Tensor,
	tensor.Tensor,
	tensor.Tensor,
	tensor.Tensor,
) error {
	return tensor.ErrNeedsPlatformSetup
}

func runMetalInstanceNorm(
	tensor.Tensor,
	tensor.Tensor,
	tensor.Tensor,
	tensor.Tensor,
) error {
	return tensor.ErrNeedsPlatformSetup
}

func runMetalBatchNormEval(
	tensor.Tensor,
	tensor.Tensor,
	tensor.Tensor,
	tensor.Tensor,
	tensor.Tensor,
	tensor.Tensor,
) error {
	return tensor.ErrNeedsPlatformSetup
}
