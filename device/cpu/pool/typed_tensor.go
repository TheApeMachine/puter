package pool

import (
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func densePointer(value tensor.Tensor) (unsafe.Pointer, dtype.DType, error) {
	format := value.DType()

	switch format {
	case dtype.Float32:
		view, err := value.Float32Native()

		if err != nil || len(view) == 0 {
			return nil, format, err
		}

		return unsafe.Pointer(unsafe.SliceData(view)), format, nil
	case dtype.BFloat16:
		view, err := value.BFloat16Native()

		if err != nil || len(view) == 0 {
			return nil, format, err
		}

		return unsafe.Pointer(unsafe.SliceData(view)), format, nil
	case dtype.Float16:
		view, err := value.Float16Native()

		if err != nil || len(view) == 0 {
			return nil, format, err
		}

		return unsafe.Pointer(unsafe.SliceData(view)), format, nil
	default:
		return nil, format, tensor.ErrDTypeMismatch
	}
}

func pool2DTyped(
	config PoolConfig,
	format dtype.DType,
	input, output tensor.Tensor,
	useMax bool,
) error {
	inputDims := input.Shape().Dims()
	outputDims := output.Shape().Dims()

	if len(inputDims) != 4 || len(outputDims) != 4 {
		return tensor.ErrShapeMismatch
	}

	batch := inputDims[0]
	channels := inputDims[1]
	inHeight := inputDims[2]
	inWidth := inputDims[3]
	outHeight := outputDims[2]
	outWidth := outputDims[3]

	if outputDims[0] != batch || outputDims[1] != channels {
		return tensor.ErrShapeMismatch
	}

	inputPointer, inputFormat, err := densePointer(input)

	if err != nil {
		return err
	}

	outputPointer, outputFormat, err := densePointer(output)

	if err != nil {
		return err
	}

	if inputFormat != format || outputFormat != format {
		return tensor.ErrDTypeMismatch
	}

	Pool2DTypedScalar(
		format,
		config,
		inputPointer, outputPointer,
		batch, channels, inHeight, inWidth, outHeight, outWidth,
		useMax,
	)

	return nil
}

func adaptivePool2DTyped(
	format dtype.DType,
	input, output tensor.Tensor,
	useMax bool,
) error {
	inputDims := input.Shape().Dims()
	outputDims := output.Shape().Dims()

	if len(inputDims) != 4 || len(outputDims) != 4 {
		return tensor.ErrShapeMismatch
	}

	batch := inputDims[0]
	channels := inputDims[1]
	inHeight := inputDims[2]
	inWidth := inputDims[3]
	outHeight := outputDims[2]
	outWidth := outputDims[3]

	if outputDims[0] != batch || outputDims[1] != channels {
		return tensor.ErrShapeMismatch
	}

	inputPointer, inputFormat, err := densePointer(input)

	if err != nil {
		return err
	}

	outputPointer, outputFormat, err := densePointer(output)

	if err != nil {
		return err
	}

	if inputFormat != format || outputFormat != format {
		return tensor.ErrDTypeMismatch
	}

	AdaptivePool2DTypedScalar(
		format,
		inputPointer, outputPointer,
		batch, channels, inHeight, inWidth, outHeight, outWidth,
		useMax,
	)

	return nil
}

func RunMaxPool2DBFloat16(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return pool2DTyped(DefaultPoolConfig(), dtype.BFloat16, args[0], args[1], true)
}

func RunMaxPool2DFloat16(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return pool2DTyped(DefaultPoolConfig(), dtype.Float16, args[0], args[1], true)
}

func RunAvgPool2DBFloat16(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return pool2DTyped(DefaultPoolConfig(), dtype.BFloat16, args[0], args[1], false)
}

func RunAvgPool2DFloat16(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return pool2DTyped(DefaultPoolConfig(), dtype.Float16, args[0], args[1], false)
}

func RunAdaptiveMaxPool2DBFloat16(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return adaptivePool2DTyped(dtype.BFloat16, args[0], args[1], true)
}

func RunAdaptiveMaxPool2DFloat16(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return adaptivePool2DTyped(dtype.Float16, args[0], args[1], true)
}

func RunAdaptiveAvgPool2DBFloat16(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return adaptivePool2DTyped(dtype.BFloat16, args[0], args[1], false)
}

func RunAdaptiveAvgPool2DFloat16(args ...tensor.Tensor) error {
	if len(args) != 2 {
		return tensor.ErrShapeMismatch
	}

	return adaptivePool2DTyped(dtype.Float16, args[0], args[1], false)
}
