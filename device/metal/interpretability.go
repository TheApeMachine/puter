package metal

import (
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/kernels"
)

var metalActivationSteerStorageDTypes = []dtype.DType{
	dtype.Float32,
	dtype.Float16,
	dtype.BFloat16,
}

func init() {
	for _, storageDType := range metalActivationSteerStorageDTypes {
		registerMetalActivationSteerKernel(storageDType)
	}
}

func registerMetalActivationSteerKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "activation_steer",
		Signature: kernels.Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				storageDType,
				storageDType,
				dtype.Float32,
			},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalActivationSteerKernel,
	})
}

func runMetalActivationSteerKernel(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return runMetalActivationSteer(args[0], args[1], args[2], args[3])
}
