//go:build !darwin || !cgo

package metal

import "github.com/theapemachine/manifesto/tensor"

func runMetalMatMul(tensor.Tensor, tensor.Tensor, tensor.Tensor) error {
	return tensor.ErrNeedsPlatformSetup
}

func runMetalMatMulAdd(
	tensor.Tensor,
	tensor.Tensor,
	tensor.Tensor,
	tensor.Tensor,
) error {
	return tensor.ErrNeedsPlatformSetup
}
