package metal

import (
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/kernels"
)

var metalDropoutDTypes = []dtype.DType{
	dtype.Float32,
	dtype.Float16,
	dtype.BFloat16,
}

func init() {
	for _, storageDType := range metalDropoutDTypes {
		registerMetalDropoutKernel(storageDType)
	}
}

func registerMetalDropoutKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "dropout",
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storageDType},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalDropoutKernel,
	})
}

func runMetalDropoutKernel(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return runMetalDropout(args[0], args[1])
}
