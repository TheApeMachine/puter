//go:build cuda

package cuda

import (
	"github.com/theapemachine/manifesto/tensor"
)

func tensorErrFromStatus(status C.CUDAStatus) error {
	switch status.code {
	case -2:
		return tensor.ErrInvalidTensor
	default:
		return tensor.ErrNeedsPlatformSetup
	}
}
