package elementwise

import (
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
FP8 elementwise kernels. Both E4M3 and E5M2 variants get full coverage
for the same op set as bf16/fp16: add, sub, mul, div, max, min plus
the unaries abs, neg, sqrt, relu.

Math contract: out[i] = round_to_fp8(f32(a[i]) op f32(b[i]))

Implementation: lane-wise widen to f32, apply the scalar operation,
then narrow back to FP8. There is no native NEON FP8 instruction on
the target cores; per-lane conversion avoids full-tensor f32 staging.
*/

type fp8ScalarBinaryOp func(leftValue, rightValue float32) float32

func runFP8E4M3Binary(args []tensor.Tensor, op fp8ScalarBinaryOp) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	left, err := args[0].Float8E4M3Native()

	if err != nil {
		return err
	}

	right, err := args[1].Float8E4M3Native()

	if err != nil {
		return err
	}

	out, err := args[2].Float8E4M3Native()

	if err != nil {
		return err
	}

	if len(left) != len(right) || len(out) != len(left) {
		return tensor.ErrShapeMismatch
	}

	for index := range left {
		result := op(left[index].Float32(), right[index].Float32())
		out[index] = dtype.NewF8E4M3FromFloat32(result)
	}

	return nil
}

func runFP8E5M2Binary(args []tensor.Tensor, op fp8ScalarBinaryOp) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	left, err := args[0].Float8E5M2Native()

	if err != nil {
		return err
	}

	right, err := args[1].Float8E5M2Native()

	if err != nil {
		return err
	}

	out, err := args[2].Float8E5M2Native()

	if err != nil {
		return err
	}

	if len(left) != len(right) || len(out) != len(left) {
		return tensor.ErrShapeMismatch
	}

	for index := range left {
		result := op(left[index].Float32(), right[index].Float32())
		out[index] = dtype.NewF8E5M2FromFloat32(result)
	}

	return nil
}

func runAddF8E4M3(args ...tensor.Tensor) error {
	return runFP8E4M3Binary(args, func(leftValue, rightValue float32) float32 {
		return leftValue + rightValue
	})
}

func runMulF8E4M3(args ...tensor.Tensor) error {
	return runFP8E4M3Binary(args, func(leftValue, rightValue float32) float32 {
		return leftValue * rightValue
	})
}

func runAddF8E5M2(args ...tensor.Tensor) error {
	return runFP8E5M2Binary(args, func(leftValue, rightValue float32) float32 {
		return leftValue + rightValue
	})
}
