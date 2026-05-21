//go:build !darwin || !cgo

package metal

import "github.com/theapemachine/manifesto/tensor"

func runMetalSoftmax(tensor.Tensor, tensor.Tensor) error {
	return tensor.ErrNeedsPlatformSetup
}
