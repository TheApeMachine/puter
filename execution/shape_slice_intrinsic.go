package execution

import (
	"fmt"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

type sliceDevice interface {
	IntrinsicSlice(
		input, output unsafe.Pointer,
		sliceLen, inputDimSize, innerBytes, start, outBytes int,
		format dtype.DType,
	)
}

type sliceLayout struct {
	dimensions   []int
	axis         int
	start        int
	end          int
	inputDimSize int
	inner        int
	outer        int
}

func runSliceIntrinsic(resolver *bindResolver) (any, error) {
	input, err := resolver.resolveInputTensor("0")

	if err != nil {
		return nil, err
	}

	layout, err := resolver.resolveSliceLayout(input)

	if err != nil {
		return nil, err
	}

	output, err := resolver.allocateOutput()

	if err != nil {
		return nil, err
	}

	if input.Location() == tensor.Host {
		return output, copySliceHost(output, input, layout)
	}

	if input.Location() != output.Location() {
		return nil, fmt.Errorf(
			"shape.slice location mismatch: input %s, output %s",
			input.Location(),
			output.Location(),
		)
	}

	return output, runSliceDeviceIntrinsic(resolver, output, input, layout)
}

func (resolver *bindResolver) resolveSliceOutputShape() (tensor.Shape, error) {
	input, err := resolver.resolveInputTensor("0")

	if err != nil {
		return tensor.Shape{}, err
	}

	layout, err := resolver.resolveSliceLayout(input)

	if err != nil {
		return tensor.Shape{}, err
	}

	return tensor.NewShape(layout.outputDimensions())
}

func (resolver *bindResolver) resolveSliceLayout(input tensor.Tensor) (sliceLayout, error) {
	dimensions, err := resolver.resolveInputDimensions("0", input)

	if err != nil {
		return sliceLayout{}, err
	}

	axis, err := sliceAxis(configInt(resolver.node, "dim", 0), dimensions)

	if err != nil {
		return sliceLayout{}, err
	}

	start := configInt(resolver.node, "start", 0)
	end := configInt(resolver.node, "end", 0)
	layout, err := newSliceLayout(dimensions, axis, start, end)

	if err != nil {
		return sliceLayout{}, err
	}

	return layout, nil
}

func sliceAxis(rawAxis int, dimensions []int) (int, error) {
	if len(dimensions) == 0 {
		return 0, fmt.Errorf("shape.slice input must have rank >= 1")
	}

	axis := rawAxis

	if axis < 0 {
		axis += len(dimensions)
	}

	if axis < 0 || axis >= len(dimensions) {
		return 0, fmt.Errorf("shape.slice dim %d out of range for shape %v", rawAxis, dimensions)
	}

	return axis, nil
}

func newSliceLayout(dimensions []int, axis, start, end int) (sliceLayout, error) {
	dimSize := dimensions[axis]
	sliceEnd := end

	if sliceEnd == 0 {
		sliceEnd = dimSize
	}

	if start < 0 || sliceEnd < start || sliceEnd > dimSize {
		return sliceLayout{}, fmt.Errorf(
			"shape.slice range [%d:%d) out of bounds for dim %d size %d",
			start,
			sliceEnd,
			axis,
			dimSize,
		)
	}

	return sliceLayout{
		dimensions:   append([]int(nil), dimensions...),
		axis:         axis,
		start:        start,
		end:          sliceEnd,
		inputDimSize: dimSize,
		inner:        productInts(dimensions[axis+1:]),
		outer:        productInts(dimensions[:axis]),
	}, nil
}

func (layout sliceLayout) sliceLen() int {
	return layout.end - layout.start
}

func (layout sliceLayout) blockElements() int {
	return layout.sliceLen() * layout.inner
}

func (layout sliceLayout) inputStrideElements() int {
	return layout.inputDimSize * layout.inner
}

func (layout sliceLayout) outputDimensions() []int {
	outputDimensions := append([]int(nil), layout.dimensions...)
	outputDimensions[layout.axis] = layout.sliceLen()

	return outputDimensions
}

func copySliceHost(output tensor.Tensor, input tensor.Tensor, layout sliceLayout) error {
	if output.DType() != input.DType() {
		return fmt.Errorf(
			"shape.slice dtype mismatch: input %s, output %s",
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
	blockBytes := layout.blockElements() * elementSize
	inputStrideBytes := layout.inputStrideElements() * elementSize
	startBytes := layout.start * layout.inner * elementSize

	if len(outputBytes) != layout.outer*blockBytes {
		return fmt.Errorf("shape.slice output byte length mismatch")
	}

	for outerIndex := range layout.outer {
		inputBase := outerIndex*inputStrideBytes + startBytes
		outputBase := outerIndex * blockBytes

		copy(outputBytes[outputBase:outputBase+blockBytes], inputBytes[inputBase:inputBase+blockBytes])
	}

	return nil
}

func runSliceDeviceIntrinsic(
	resolver *bindResolver,
	output tensor.Tensor,
	input tensor.Tensor,
	layout sliceLayout,
) error {
	deviceBackend, ok := resolver.dispatcher.deviceBackend.(sliceDevice)

	if !ok {
		return fmt.Errorf(
			"shape.slice: backend %T cannot run %s tensor",
			resolver.dispatcher.deviceBackend,
			input.Location(),
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

	deviceBackend.IntrinsicSlice(
		inputPointer,
		outputPointer,
		layout.sliceLen(),
		layout.inputDimSize,
		layout.inner*elementSize,
		layout.start,
		output.Bytes(),
		input.DType(),
	)

	return nil
}
