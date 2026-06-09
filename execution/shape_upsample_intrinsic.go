package execution

import (
	"fmt"

	"github.com/theapemachine/manifesto/tensor"
)

type upsampleNearest2DLayout struct {
	inputDimensions  []int
	outputDimensions []int
	channels         int
	inHeight         int
	inWidth          int
	outHeight        int
	outWidth         int
	outElements      int
}

func runUpsampleNearest2DIntrinsic(resolver *bindResolver) (any, error) {
	input, err := resolver.resolveInputTensor("0")

	if err != nil {
		return nil, err
	}

	layout, err := resolver.resolveUpsampleNearest2DLayout(input)

	if err != nil {
		return nil, err
	}

	output, err := resolver.allocateOutput()

	if err != nil {
		return nil, err
	}

	if input.Location() == tensor.Host {
		return output, copyUpsampleNearest2DHost(output, input, layout)
	}

	if input.Location() != output.Location() {
		return nil, fmt.Errorf(
			"shape.upsample_nearest2d location mismatch: input %s, output %s",
			input.Location(),
			output.Location(),
		)
	}

	return output, runUpsampleNearest2DDeviceIntrinsic(resolver, output, input, layout)
}

func (resolver *bindResolver) resolveUpsampleNearest2DOutputShape() (tensor.Shape, error) {
	input, err := resolver.resolveInputTensor("0")

	if err != nil {
		return tensor.Shape{}, err
	}

	layout, err := resolver.resolveUpsampleNearest2DLayout(input)

	if err != nil {
		return tensor.Shape{}, err
	}

	return tensor.NewShape(layout.outputDimensions)
}

func (resolver *bindResolver) resolveUpsampleNearest2DLayout(input tensor.Tensor) (upsampleNearest2DLayout, error) {
	dimensions, err := resolver.resolveInputDimensions("0", input)

	if err != nil {
		return upsampleNearest2DLayout{}, err
	}

	if len(dimensions) != 4 {
		return upsampleNearest2DLayout{}, fmt.Errorf(
			"shape.upsample_nearest2d input must be rank 4, got %d",
			len(dimensions),
		)
	}

	scaleH := configInt(resolver.node, "scale_h", 0)
	scaleW := configInt(resolver.node, "scale_w", 0)

	if scaleH <= 0 || scaleW <= 0 {
		return upsampleNearest2DLayout{}, fmt.Errorf("shape.upsample_nearest2d requires positive scales")
	}

	outputDimensions := append([]int(nil), dimensions...)
	outputDimensions[2] = dimensions[2] * scaleH
	outputDimensions[3] = dimensions[3] * scaleW

	return upsampleNearest2DLayout{
		inputDimensions:  append([]int(nil), dimensions...),
		outputDimensions: outputDimensions,
		channels:         dimensions[1],
		inHeight:         dimensions[2],
		inWidth:          dimensions[3],
		outHeight:        outputDimensions[2],
		outWidth:         outputDimensions[3],
		outElements:      productInts(outputDimensions),
	}, nil
}

func copyUpsampleNearest2DHost(
	output tensor.Tensor,
	input tensor.Tensor,
	layout upsampleNearest2DLayout,
) error {
	if output.DType() != input.DType() {
		return fmt.Errorf(
			"shape.upsample_nearest2d dtype mismatch: input %s, output %s",
			input.DType(),
			output.DType(),
		)
	}

	elementSize, err := input.DType().Size()

	if err != nil {
		return err
	}

	inputPointer, _, err := pointerOf(input)

	if err != nil {
		return err
	}

	outputPointer, _, err := pointerOf(output)

	if err != nil {
		return err
	}

	inputBytes := hostByteSlice(inputPointer, input.Bytes())
	outputBytes := hostByteSlice(outputPointer, output.Bytes())

	for outputIndex := range layout.outElements {
		inputIndex := upsampleNearest2DInputIndex(layout, outputIndex)
		copyElementBytes(outputBytes, inputBytes, outputIndex, inputIndex, elementSize)
	}

	return nil
}

func upsampleNearest2DInputIndex(layout upsampleNearest2DLayout, outputIndex int) int {
	outCol := outputIndex % layout.outWidth
	outRow := (outputIndex / layout.outWidth) % layout.outHeight
	channel := (outputIndex / (layout.outWidth * layout.outHeight)) % layout.channels
	batch := outputIndex / (layout.outWidth * layout.outHeight * layout.channels)
	inRow := outRow * layout.inHeight / layout.outHeight
	inCol := outCol * layout.inWidth / layout.outWidth

	return ((batch*layout.channels+channel)*layout.inHeight+inRow)*layout.inWidth + inCol
}

func copyElementBytes(outputBytes, inputBytes []byte, outputIndex, inputIndex, elementSize int) {
	outputOffset := outputIndex * elementSize
	inputOffset := inputIndex * elementSize

	copy(outputBytes[outputOffset:outputOffset+elementSize], inputBytes[inputOffset:inputOffset+elementSize])
}

func runUpsampleNearest2DDeviceIntrinsic(
	resolver *bindResolver,
	output tensor.Tensor,
	input tensor.Tensor,
	layout upsampleNearest2DLayout,
) error {
	deviceBackend, ok := resolver.dispatcher.deviceBackend.(upsampleNearest2DDevice)

	if !ok {
		return fmt.Errorf(
			"shape.upsample_nearest2d backend %T does not implement UpsampleNearest2D",
			resolver.dispatcher.deviceBackend,
		)
	}

	inputPointer, _, err := pointerOf(input)

	if err != nil {
		return err
	}

	outputPointer, _, err := pointerOf(output)

	if err != nil {
		return err
	}

	deviceBackend.IntrinsicUpsampleNearest2D(
		inputPointer,
		outputPointer,
		layout.channels,
		layout.inHeight,
		layout.inWidth,
		layout.outHeight,
		layout.outWidth,
		layout.outElements,
		input.DType(),
	)

	return nil
}
