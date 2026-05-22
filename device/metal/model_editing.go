package metal

import (
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/kernels"
)

var metalWeightGraftStorageDTypes = []dtype.DType{
	dtype.Float32,
	dtype.Float16,
	dtype.BFloat16,
}

func init() {
	for _, storageDType := range metalWeightGraftStorageDTypes {
		registerMetalWeightGraftAddKernel(storageDType)
	}
}

func registerMetalWeightGraftAddKernel(storageDType dtype.DType) {
	kernels.Default.Register(kernels.Kernel{
		Name: "weight_graft_add",
		Signature: kernels.Signature{
			Layout: tensor.LayoutDense,
			Inputs: []dtype.DType{
				storageDType,
				storageDType,
			},
			Outputs: []dtype.DType{storageDType},
		},
		Locations: []tensor.Location{tensor.Metal},
		Run:       runMetalWeightGraftAddKernel,
	})
}

func runMetalWeightGraftAddKernel(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return runMetalWeightGraftAdd(args[0], args[1])
}
