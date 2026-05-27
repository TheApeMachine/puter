package execution

import (
	"fmt"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

type transposeDevice interface {
	Transpose(
		input, output unsafe.Pointer,
		rank, count int,
		permutation, inputStrides, outputStrides []uint32,
		format dtype.DType,
	)
}

type transposeLayout struct {
	inputDimensions  []int
	outputDimensions []int
	permutation      []uint32
	inputStrides     []uint32
	outputStrides    []uint32
	count            int
}

func runTransposeIntrinsic(resolver *bindResolver) (any, error) {
	input, err := resolver.resolveInputTensor("0")

	if err != nil {
		return nil, err
	}

	liveInput, err := resolver.liveInputTensor("0", input)

	if err != nil {
		return nil, err
	}

	layout, err := resolver.resolveTransposeLayout(liveInput)

	if err != nil {
		return nil, err
	}

	output, err := resolver.allocateOutput()

	if err != nil {
		return nil, err
	}

	if liveInput.Location() == tensor.Host {
		return output, copyTransposeHost(output, liveInput, layout)
	}

	if liveInput.Location() != output.Location() {
		return nil, fmt.Errorf(
			"shape.transpose location mismatch: input %s, output %s",
			liveInput.Location(),
			output.Location(),
		)
	}

	return output, runTransposeDeviceIntrinsic(resolver, output, liveInput, layout)
}

func (resolver *bindResolver) resolveTransposeOutputShape() (tensor.Shape, error) {
	input, err := resolver.resolveInputTensor("0")

	if err != nil {
		return tensor.Shape{}, err
	}

	layout, err := resolver.resolveTransposeLayout(input)

	if err != nil {
		return tensor.Shape{}, err
	}

	return tensor.NewShape(layout.outputDimensions)
}

func (resolver *bindResolver) resolveTransposeLayout(input tensor.Tensor) (transposeLayout, error) {
	dimensions, err := resolver.resolveInputDimensions("0", input)

	if err != nil {
		return transposeLayout{}, err
	}

	firstAxis, err := transposeAxis(configInt(resolver.node, "dim0", 0), dimensions)

	if err != nil {
		return transposeLayout{}, err
	}

	secondAxis, err := transposeAxis(configInt(resolver.node, "dim1", 1), dimensions)

	if err != nil {
		return transposeLayout{}, err
	}

	return newTransposeLayout(dimensions, firstAxis, secondAxis)
}

func transposeAxis(rawAxis int, dimensions []int) (int, error) {
	if len(dimensions) == 0 {
		return 0, fmt.Errorf("shape.transpose input must have rank >= 1")
	}

	axis := rawAxis

	if axis < 0 {
		axis += len(dimensions)
	}

	if axis < 0 || axis >= len(dimensions) {
		return 0, fmt.Errorf("shape.transpose dim %d out of range for shape %v", rawAxis, dimensions)
	}

	return axis, nil
}

func newTransposeLayout(dimensions []int, firstAxis, secondAxis int) (transposeLayout, error) {
	outputDimensions := append([]int(nil), dimensions...)
	outputDimensions[firstAxis], outputDimensions[secondAxis] =
		outputDimensions[secondAxis], outputDimensions[firstAxis]

	permutation := make([]uint32, len(dimensions))

	for axis := range dimensions {
		permutation[axis] = uint32(axis)
	}

	permutation[firstAxis], permutation[secondAxis] =
		permutation[secondAxis], permutation[firstAxis]

	return transposeLayout{
		inputDimensions:  append([]int(nil), dimensions...),
		outputDimensions: outputDimensions,
		permutation:      permutation,
		inputStrides:     uint32Strides(dimensions),
		outputStrides:    uint32Strides(outputDimensions),
		count:            productInts(dimensions),
	}, nil
}

func uint32Strides(dimensions []int) []uint32 {
	strides := make([]uint32, len(dimensions))

	if len(dimensions) == 0 {
		return strides
	}

	strides[len(dimensions)-1] = 1

	for index := len(dimensions) - 2; index >= 0; index-- {
		strides[index] = strides[index+1] * uint32(dimensions[index+1])
	}

	return strides
}

func copyTransposeHost(output tensor.Tensor, input tensor.Tensor, layout transposeLayout) error {
	if output.DType() != input.DType() {
		return fmt.Errorf(
			"shape.transpose dtype mismatch: input %s, output %s",
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

	inputBytes := unsafe.Slice((*byte)(inputPointer), input.Bytes())
	outputBytes := unsafe.Slice((*byte)(outputPointer), output.Bytes())
	inputCoordinates := make([]int, len(layout.inputDimensions))
	outputCoordinates := make([]int, len(layout.outputDimensions))

	for inputFlat := range layout.count {
		fillCoordinates(inputFlat, layout.inputStrides, inputCoordinates)

		for outputAxis, inputAxis := range layout.permutation {
			outputCoordinates[outputAxis] = inputCoordinates[int(inputAxis)]
		}

		outputFlat := flattenCoordinates(outputCoordinates, layout.outputStrides)
		copyElement(outputBytes, inputBytes, outputFlat, inputFlat, elementSize)
	}

	return nil
}

func fillCoordinates(flatIndex int, strides []uint32, coordinates []int) {
	remainder := flatIndex

	for axis, stride := range strides {
		coordinates[axis] = remainder / int(stride)
		remainder %= int(stride)
	}
}

func flattenCoordinates(coordinates []int, strides []uint32) int {
	flatIndex := 0

	for axis, coordinate := range coordinates {
		flatIndex += coordinate * int(strides[axis])
	}

	return flatIndex
}

func copyElement(outputBytes, inputBytes []byte, outputIndex, inputIndex, elementSize int) {
	outputOffset := outputIndex * elementSize
	inputOffset := inputIndex * elementSize

	copy(outputBytes[outputOffset:outputOffset+elementSize], inputBytes[inputOffset:inputOffset+elementSize])
}

func runTransposeDeviceIntrinsic(
	resolver *bindResolver,
	output tensor.Tensor,
	input tensor.Tensor,
	layout transposeLayout,
) error {
	deviceBackend, ok := resolver.dispatcher.deviceBackend.(transposeDevice)

	if !ok {
		return fmt.Errorf(
			"shape.transpose: backend %T cannot run %s tensor",
			resolver.dispatcher.deviceBackend,
			input.Location(),
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

	deviceBackend.Transpose(
		inputPointer,
		outputPointer,
		len(layout.inputDimensions),
		layout.count,
		layout.permutation,
		layout.inputStrides,
		layout.outputStrides,
		input.DType(),
	)

	return nil
}
