package convolution

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

func conv2DTyped(
	config Conv2DConfig,
	format dtype.DType,
	input, weight, bias, output tensor.Tensor,
) error {
	inputDims := input.Shape().Dims()
	weightDims := weight.Shape().Dims()
	biasDims := bias.Shape().Dims()
	outputDims := output.Shape().Dims()

	if len(inputDims) != 4 || len(weightDims) != 4 ||
		len(biasDims) != 1 || len(outputDims) != 4 {
		return tensor.ErrShapeMismatch
	}

	batch := inputDims[0]
	inChannels := inputDims[1]
	inHeight := inputDims[2]
	inWidth := inputDims[3]

	outChannels := weightDims[0]
	kernelInChannels := weightDims[1]
	kernelHeight := weightDims[2]
	kernelWidth := weightDims[3]

	outHeight := outputDims[2]
	outWidth := outputDims[3]

	if kernelInChannels != inChannels ||
		biasDims[0] != outChannels ||
		outputDims[0] != batch ||
		outputDims[1] != outChannels {
		return tensor.ErrShapeMismatch
	}

	inputPointer, inputFormat, err := densePointer(input)

	if err != nil {
		return err
	}

	weightPointer, weightFormat, err := densePointer(weight)

	if err != nil {
		return err
	}

	biasPointer, biasFormat, err := densePointer(bias)

	if err != nil {
		return err
	}

	outputPointer, outputFormat, err := densePointer(output)

	if err != nil {
		return err
	}

	if inputFormat != format || weightFormat != format ||
		biasFormat != format || outputFormat != format {
		return tensor.ErrDTypeMismatch
	}

	Conv2D(
		config,
		inputPointer, weightPointer, biasPointer, outputPointer,
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth,
		format,
	)

	return nil
}

func conv1DTyped(
	config Conv1DConfig,
	format dtype.DType,
	input, weight, bias, output tensor.Tensor,
) error {
	inDims := input.Shape().Dims()
	weightDims := weight.Shape().Dims()
	outDims := output.Shape().Dims()

	if len(inDims) != 3 || len(weightDims) != 3 || len(outDims) != 3 {
		return tensor.ErrShapeMismatch
	}

	batch := inDims[0]
	inChannels := inDims[1]
	inLength := inDims[2]
	outChannels := weightDims[0]
	kernelLength := weightDims[2]
	outLength := outDims[2]

	inputPointer, inputFormat, err := densePointer(input)

	if err != nil {
		return err
	}

	weightPointer, weightFormat, err := densePointer(weight)

	if err != nil {
		return err
	}

	biasPointer, biasFormat, err := densePointer(bias)

	if err != nil {
		return err
	}

	outputPointer, outputFormat, err := densePointer(output)

	if err != nil {
		return err
	}

	if inputFormat != format || weightFormat != format ||
		biasFormat != format || outputFormat != format {
		return tensor.ErrDTypeMismatch
	}

	Conv1D(
		config,
		inputPointer, weightPointer, biasPointer, outputPointer,
		batch, inChannels, inLength, outChannels, kernelLength, outLength,
		format,
	)

	return nil
}

func conv3DTyped(
	config Conv3DConfig,
	format dtype.DType,
	input, weight, bias, output tensor.Tensor,
) error {
	inDims := input.Shape().Dims()
	weightDims := weight.Shape().Dims()
	outDims := output.Shape().Dims()

	if len(inDims) != 5 || len(weightDims) != 5 || len(outDims) != 5 {
		return tensor.ErrShapeMismatch
	}

	batch := inDims[0]
	inChannels := inDims[1]
	inD := inDims[2]
	inH := inDims[3]
	inW := inDims[4]

	outChannels := weightDims[0]
	kD := weightDims[2]
	kH := weightDims[3]
	kW := weightDims[4]

	outD := outDims[2]
	outH := outDims[3]
	outW := outDims[4]

	inputPointer, inputFormat, err := densePointer(input)

	if err != nil {
		return err
	}

	weightPointer, weightFormat, err := densePointer(weight)

	if err != nil {
		return err
	}

	biasPointer, biasFormat, err := densePointer(bias)

	if err != nil {
		return err
	}

	outputPointer, outputFormat, err := densePointer(output)

	if err != nil {
		return err
	}

	if inputFormat != format || weightFormat != format ||
		biasFormat != format || outputFormat != format {
		return tensor.ErrDTypeMismatch
	}

	Conv3D(
		config,
		inputPointer, weightPointer, biasPointer, outputPointer,
		batch, inChannels, inD, inH, inW,
		outChannels, kD, kH, kW, outD, outH, outW,
		format,
	)

	return nil
}

func RunConv2DBFloat16(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return conv2DTyped(DefaultConv2DConfig(), dtype.BFloat16, args[0], args[1], args[2], args[3])
}

func RunConv2DFloat16(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return conv2DTyped(DefaultConv2DConfig(), dtype.Float16, args[0], args[1], args[2], args[3])
}

func RunConv1DBFloat16(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return conv1DTyped(DefaultConv1DConfig(), dtype.BFloat16, args[0], args[1], args[2], args[3])
}

func RunConv1DFloat16(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return conv1DTyped(DefaultConv1DConfig(), dtype.Float16, args[0], args[1], args[2], args[3])
}

func RunConv3DBFloat16(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return conv3DTyped(DefaultConv3DConfig(), dtype.BFloat16, args[0], args[1], args[2], args[3])
}

func RunConv3DFloat16(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return conv3DTyped(DefaultConv3DConfig(), dtype.Float16, args[0], args[1], args[2], args[3])
}

func RunConvTranspose2DBFloat16(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return convTranspose2DTyped(DefaultConv2DConfig(), dtype.BFloat16, args[0], args[1], args[2], args[3])
}

func RunConvTranspose2DFloat16(args ...tensor.Tensor) error {
	if len(args) != 4 {
		return tensor.ErrShapeMismatch
	}

	return convTranspose2DTyped(DefaultConv2DConfig(), dtype.Float16, args[0], args[1], args[2], args[3])
}

func convTranspose2DTyped(
	config Conv2DConfig,
	format dtype.DType,
	input, weight, bias, output tensor.Tensor,
) error {
	inDims := input.Shape().Dims()
	weightDims := weight.Shape().Dims()
	outDims := output.Shape().Dims()

	if len(inDims) != 4 || len(weightDims) != 4 || len(outDims) != 4 {
		return tensor.ErrShapeMismatch
	}

	batch := inDims[0]
	inChannels := inDims[1]
	inHeight := inDims[2]
	inWidth := inDims[3]
	outChannels := weightDims[1]
	kernelHeight := weightDims[2]
	kernelWidth := weightDims[3]
	outHeight := outDims[2]
	outWidth := outDims[3]

	inputPointer, inputFormat, err := densePointer(input)

	if err != nil {
		return err
	}

	weightPointer, weightFormat, err := densePointer(weight)

	if err != nil {
		return err
	}

	biasPointer, biasFormat, err := densePointer(bias)

	if err != nil {
		return err
	}

	outputPointer, outputFormat, err := densePointer(output)

	if err != nil {
		return err
	}

	if inputFormat != format || weightFormat != format ||
		biasFormat != format || outputFormat != format {
		return tensor.ErrDTypeMismatch
	}

	ConvTranspose2D(
		config,
		inputPointer, weightPointer, biasPointer, outputPointer,
		batch, inChannels, inHeight, inWidth,
		outChannels, kernelHeight, kernelWidth,
		outHeight, outWidth,
		format,
	)

	return nil
}
