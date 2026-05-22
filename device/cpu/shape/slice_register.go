package shape

import (
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/kernels"
)

func init() {
	registerSliceReducedPrecision()
	registerPageStateKernels()
}

func registerSliceReducedPrecision() {
	for _, paramDType := range []dtype.DType{dtype.BFloat16, dtype.Float16} {
		paramDType := paramDType

		kernels.Default.Register(kernels.Kernel{
			Name: "slice",
			Signature: kernels.Signature{
				Layout: tensor.LayoutDense,
				Inputs: []dtype.DType{
					paramDType,
					dtype.Int32,
					dtype.Int32,
					dtype.Int32,
				},
				Outputs: []dtype.DType{paramDType},
			},
			Locations: []tensor.Location{tensor.Host},
			Run: func(args ...tensor.Tensor) error {
				return runSlice(args...)
			},
		})
	}
}

func registerPageStateKernels() {
	for _, paramDType := range []dtype.DType{dtype.Float32, dtype.BFloat16, dtype.Float16} {
		paramDType := paramDType

		kernels.Default.Register(kernels.Kernel{
			Name: "page_write",
			Signature: kernels.Signature{
				Layout: tensor.LayoutDense,
				Inputs: []dtype.DType{
					paramDType,
					paramDType,
					dtype.Int32,
					dtype.Int32,
					dtype.Int32,
				},
				Outputs: []dtype.DType{paramDType},
			},
			Locations: []tensor.Location{tensor.Host},
			Run: func(args ...tensor.Tensor) error {
				return runPageWrite(args...)
			},
		})

		kernels.Default.Register(kernels.Kernel{
			Name: "page_gather",
			Signature: kernels.Signature{
				Layout: tensor.LayoutDense,
				Inputs: []dtype.DType{
					paramDType,
					dtype.Int32,
					dtype.Int32,
				},
				Outputs: []dtype.DType{paramDType},
			},
			Locations: []tensor.Location{tensor.Host},
			Run: func(args ...tensor.Tensor) error {
				return runPageGather(args...)
			},
		})
	}
}
