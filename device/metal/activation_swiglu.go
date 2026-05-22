package metal

import (
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/kernels"
)

var metalSwiGLUDTypes = []dtype.DType{
	dtype.Float32,
	dtype.Float16,
	dtype.BFloat16,
}

func init() {
	for _, storageDType := range metalSwiGLUDTypes {
		registerMetalSwiGLUKernel(storageDType)
	}
}

func registerMetalSwiGLUKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "swiglu",
		Signature: kernels.Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				storageDType,
				storageDType,
			},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalSwiGLUKernel,
	})

	kernels.Default.Register(kernels.Kernel{
		Name: "swiglu",
		Signature: kernels.Signature{
			Layout:  tensor.LayoutDense,
			Inputs:  []dtype.DType{storageDType},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalPackedSwiGLUKernel,
	})
}

func runMetalSwiGLUKernel(args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	return runMetalSwiGLU(args[0], args[1], args[2])
}

func runMetalPackedSwiGLUKernel(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return runMetalPackedSwiGLU(args[0], args[1])
}
