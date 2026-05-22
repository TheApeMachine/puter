package metal

import (
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/kernels"
)

func init() {
	for _, storageDType := range metalProjectionDTypes {
		registerMetalTimestepKernel(storageDType)
	}
}

func registerMetalTimestepKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "timestep",
		Signature: kernels.Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				dtype.Float32,
				dtype.Float32,
				dtype.Float32,
				dtype.Int32,
			},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalTimestepKernel,
	})
}

func runMetalTimestepKernel(args ...tensor.Tensor) error {
	if len(args) != 5 {
		return tensor.ErrShapeMismatch
	}

	return runMetalTimestep(args[0], args[1], args[2], args[3], args[4])
}
