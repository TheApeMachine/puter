package shape

import (
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
Mixed-precision dispatch for shape-manipulation ops. These ops are
dtype-agnostic data movement (gather/scatter/transpose copy bytes;
where/masked_fill conditionally select bytes). Reduced-precision
signatures delegate directly to the shared runners, which operate on
native storage via element-sized byte copies.
*/

func runShapeUnaryMixed(
	args []tensor.Tensor,
	_ dtype.DType,
	runner func(args ...tensor.Tensor) error,
) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return runner(args...)
}

func runShapeOpWithIntIndex(
	args []tensor.Tensor,
	_ dtype.DType,
	runner func(args ...tensor.Tensor) error,
	_ int,
) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	return runner(args...)
}

func runScatterMixed(args []tensor.Tensor, _ dtype.DType) error {
	return runScatterFloat32Int32(args...)
}

func runSliceMixed(args []tensor.Tensor, _ dtype.DType) error {
	return runSlice(args...)
}

func runConcatMixed(args []tensor.Tensor, _ dtype.DType) error {
	return runConcatFloat32(args...)
}
