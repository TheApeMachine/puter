package elementwise

import (
	"fmt"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func RunAdd(args ...tensor.Tensor) error {
	return runBinaryByDType("add", args...)
}

func RunSub(args ...tensor.Tensor) error {
	return runBinaryByDType("sub", args...)
}

func RunMul(args ...tensor.Tensor) error {
	return runBinaryByDType("mul", args...)
}

func RunDiv(args ...tensor.Tensor) error {
	return runBinaryByDType("div", args...)
}

func runBinaryByDType(kernel string, args ...tensor.Tensor) error {
	if len(args) != 3 {
		return tensor.ErrShapeMismatch
	}

	storageDType := args[2].DType()

	switch storageDType {
	case dtype.Float32:
		return runBinaryFloat32(kernel, args...)
	case dtype.Float16:
		return runBinaryFloat16(kernel, args...)
	case dtype.BFloat16:
		return runBinaryBFloat16(kernel, args...)
	default:
		return fmt.Errorf("elementwise: unsupported binary dtype %s", storageDType)
	}
}

func runBinaryFloat32(kernel string, args ...tensor.Tensor) error {
	left, right, out, err := binaryFloat32Args(args)

	if err != nil {
		return err
	}

	switch kernel {
	case "add":
		AddFloat32Native(out, left, right)
	case "sub":
		SubFloat32Native(out, left, right)
	case "mul":
		MulFloat32Native(out, left, right)
	case "div":
		DivFloat32Native(out, left, right)
	default:
		return fmt.Errorf("elementwise: unsupported float32 binary kernel %q", kernel)
	}

	return nil
}

func runBinaryFloat16(kernel string, args ...tensor.Tensor) error {
	left, right, out, err := binaryFloat16Args(args)

	if err != nil {
		return err
	}

	switch kernel {
	case "add":
		AddFloat16Native(out, left, right)
	case "sub":
		SubFloat16Native(out, left, right)
	case "mul":
		MulFloat16Native(out, left, right)
	case "div":
		DivFloat16Native(out, left, right)
	default:
		return fmt.Errorf("elementwise: unsupported float16 binary kernel %q", kernel)
	}

	return nil
}

func runBinaryBFloat16(kernel string, args ...tensor.Tensor) error {
	left, right, out, err := binaryBFloat16Args(args)

	if err != nil {
		return err
	}

	switch kernel {
	case "add":
		AddBFloat16Native(out, left, right)
	case "sub":
		SubBFloat16Native(out, left, right)
	case "mul":
		MulBFloat16Native(out, left, right)
	case "div":
		DivBFloat16Native(out, left, right)
	default:
		return fmt.Errorf("elementwise: unsupported bfloat16 binary kernel %q", kernel)
	}

	return nil
}

func binaryFloat32Args(args []tensor.Tensor) (left, right, out []float32, err error) {
	if len(args) != 3 {
		return nil, nil, nil, tensor.ErrShapeMismatch
	}

	left, err = args[0].Float32Native()

	if err != nil {
		return nil, nil, nil, err
	}

	right, err = args[1].Float32Native()

	if err != nil {
		return nil, nil, nil, err
	}

	out, err = args[2].Float32Native()

	if err != nil {
		return nil, nil, nil, err
	}

	if len(left) != len(out) || len(right) != len(out) {
		return nil, nil, nil, tensor.ErrShapeMismatch
	}

	return left, right, out, nil
}

func binaryFloat16Args(args []tensor.Tensor) (left, right, out []dtype.F16, err error) {
	if len(args) != 3 {
		return nil, nil, nil, tensor.ErrShapeMismatch
	}

	left, err = args[0].Float16Native()

	if err != nil {
		return nil, nil, nil, err
	}

	right, err = args[1].Float16Native()

	if err != nil {
		return nil, nil, nil, err
	}

	out, err = args[2].Float16Native()

	if err != nil {
		return nil, nil, nil, err
	}

	if len(left) != len(out) || len(right) != len(out) {
		return nil, nil, nil, tensor.ErrShapeMismatch
	}

	return left, right, out, nil
}

func binaryBFloat16Args(args []tensor.Tensor) (left, right, out []dtype.BF16, err error) {
	if len(args) != 3 {
		return nil, nil, nil, tensor.ErrShapeMismatch
	}

	left, err = args[0].BFloat16Native()

	if err != nil {
		return nil, nil, nil, err
	}

	right, err = args[1].BFloat16Native()

	if err != nil {
		return nil, nil, nil, err
	}

	out, err = args[2].BFloat16Native()

	if err != nil {
		return nil, nil, nil, err
	}

	if len(left) != len(out) || len(right) != len(out) {
		return nil, nil, nil, tensor.ErrShapeMismatch
	}

	return left, right, out, nil
}
