package execution

import (
	"fmt"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

type lastTokenDevice interface {
	LastToken(
		input, output unsafe.Pointer,
		seq, hiddenBytes, outBytes int,
		format dtype.DType,
	)
}

type concatDevice interface {
	Concat(
		left, right, output unsafe.Pointer,
		leftBytes, rightBytes int,
		format dtype.DType,
	)
	ConcatLastDim(
		left, right, output unsafe.Pointer,
		leftRowBytes, rightRowBytes, rowBytes, totalBytes int,
		format dtype.DType,
	)
}

func isIntrinsicMethod(method string) bool {
	switch method {
	case "shape.view_as_heads", "shape.merge_heads", "shape.last_token", "shape.concat":
		return true
	default:
		return false
	}
}

func runIntrinsic(resolver *bindResolver) (any, error) {
	switch resolver.bind.Method {
	case "shape.view_as_heads", "shape.merge_heads":
		return runReshapeIntrinsic(resolver)
	case "shape.last_token":
		return runLastTokenIntrinsic(resolver)
	case "shape.concat":
		return runConcatIntrinsic(resolver)
	default:
		return nil, fmt.Errorf("unknown intrinsic %q", resolver.bind.Method)
	}
}

func runReshapeIntrinsic(resolver *bindResolver) (any, error) {
	input, err := resolver.resolveInputTensor("0")

	if err != nil {
		return nil, err
	}

	if input.Len() != resolver.outputShape.Len() {
		return nil, fmt.Errorf(
			"reshape element count mismatch: input %d, output %d",
			input.Len(), resolver.outputShape.Len(),
		)
	}

	return input.Reshape(resolver.outputShape.Dims())
}

func runLastTokenIntrinsic(resolver *bindResolver) (any, error) {
	input, err := resolver.resolveInputTensor("0")

	if err != nil {
		return nil, err
	}

	dimensions := substituteLaunchDimensions(
		input.Shape().Dims(),
		resolver.dispatcher.maxBindings,
		resolver.dispatcher.launchBindings,
	)

	if len(dimensions) < 2 {
		return nil, fmt.Errorf("shape.last_token input must have rank >= 2, got %d", len(dimensions))
	}

	batch, sequence, rowElements := lastTokenLayout(dimensions)

	if batch < 1 || sequence < 1 {
		return nil, fmt.Errorf("shape.last_token input has empty batch or sequence")
	}

	if input.Location() != tensor.Host {
		return runLastTokenDeviceIntrinsic(resolver, input, batch, sequence, rowElements)
	}

	output, err := resolver.allocateOutput()

	if err != nil {
		return nil, err
	}

	if err := copyLastTokenHost(output, input, batch, sequence, rowElements); err != nil {
		return nil, err
	}

	return output, nil
}

func (resolver *bindResolver) resolveLastTokenOutputShape() (tensor.Shape, error) {
	input, err := resolver.resolveInputTensor("0")

	if err != nil {
		return tensor.Shape{}, err
	}

	dimensions := substituteLaunchDimensions(
		input.Shape().Dims(),
		resolver.dispatcher.maxBindings,
		resolver.dispatcher.launchBindings,
	)

	if len(dimensions) < 2 {
		return tensor.Shape{}, fmt.Errorf("shape.last_token input must have rank >= 2, got %d", len(dimensions))
	}

	batch, sequence, _ := lastTokenLayout(dimensions)

	if batch < 1 || sequence < 1 {
		return tensor.Shape{}, fmt.Errorf("shape.last_token input has empty batch or sequence")
	}

	if len(dimensions) == 2 {
		return tensor.NewShape([]int{1, dimensions[1]})
	}

	outputDimensions := append([]int{dimensions[0]}, dimensions[2:]...)

	return tensor.NewShape(outputDimensions)
}

func lastTokenLayout(dimensions []int) (int, int, int) {
	if len(dimensions) == 2 {
		return 1, dimensions[0], dimensions[1]
	}

	return dimensions[0], dimensions[1], productInts(dimensions[2:])
}

func copyLastTokenHost(
	output tensor.Tensor,
	input tensor.Tensor,
	batch int,
	sequence int,
	rowElements int,
) error {
	for batchIndex := range batch {
		start := (batchIndex*sequence + sequence - 1) * rowElements
		slice, err := input.Slice(start, rowElements)

		if err != nil {
			return err
		}

		outputSlice, err := output.Slice(batchIndex*rowElements, rowElements)

		if err != nil {
			return err
		}

		if err := copyTensorStorage(outputSlice, slice, rowElements); err != nil {
			return err
		}
	}

	return nil
}

func runConcatIntrinsic(resolver *bindResolver) (any, error) {
	left, right, axis, err := resolver.resolveConcatInputs()

	if err != nil {
		return nil, err
	}

	output, err := resolver.allocateOutput()

	if err != nil {
		return nil, err
	}

	if left.Location() == tensor.Host && right.Location() == tensor.Host {
		return output, copyConcatHost(output, left, right, axis)
	}

	if left.Location() != right.Location() || output.Location() != left.Location() {
		return nil, fmt.Errorf(
			"shape.concat location mismatch: left %s, right %s, output %s",
			left.Location(), right.Location(), output.Location(),
		)
	}

	return output, runConcatDeviceIntrinsic(resolver, output, left, right, axis)
}

func (resolver *bindResolver) resolveConcatOutputShape() (tensor.Shape, error) {
	left, right, axis, err := resolver.resolveConcatInputs()

	if err != nil {
		return tensor.Shape{}, err
	}

	outputDimensions := left.Shape().Dims()
	rightDimensions := right.Shape().Dims()
	outputDimensions[axis] += rightDimensions[axis]

	return tensor.NewShape(outputDimensions)
}

func (resolver *bindResolver) resolveConcatInputs() (tensor.Tensor, tensor.Tensor, int, error) {
	left, err := resolver.resolveInputTensor("0")

	if err != nil {
		return nil, nil, 0, err
	}

	right, err := resolver.resolveInputTensor("1")

	if err != nil {
		return nil, nil, 0, err
	}

	if left.DType() != right.DType() {
		return nil, nil, 0, fmt.Errorf(
			"shape.concat dtype mismatch: left %s, right %s",
			left.DType(), right.DType(),
		)
	}

	axis, err := concatAxis(configInt(resolver.node, "dim", 0), left.Shape().Dims(), right.Shape().Dims())

	if err != nil {
		return nil, nil, 0, err
	}

	return left, right, axis, nil
}

func concatAxis(rawAxis int, leftDimensions []int, rightDimensions []int) (int, error) {
	if len(leftDimensions) != len(rightDimensions) {
		return 0, fmt.Errorf(
			"shape.concat rank mismatch: left %d, right %d",
			len(leftDimensions), len(rightDimensions),
		)
	}

	if len(leftDimensions) == 0 {
		return 0, fmt.Errorf("shape.concat input must have rank >= 1")
	}

	axis := rawAxis

	if axis < 0 {
		axis += len(leftDimensions)
	}

	if axis < 0 || axis >= len(leftDimensions) {
		return 0, fmt.Errorf("shape.concat dim %d out of range for shape %v", rawAxis, leftDimensions)
	}

	for index := range leftDimensions {
		if index == axis {
			continue
		}

		if leftDimensions[index] != rightDimensions[index] {
			return 0, fmt.Errorf(
				"shape.concat dim %d mismatch: left %d, right %d",
				index, leftDimensions[index], rightDimensions[index],
			)
		}
	}

	return axis, nil
}

func copyConcatHost(output tensor.Tensor, left tensor.Tensor, right tensor.Tensor, axis int) error {
	elementSize, err := left.DType().Size()

	if err != nil {
		return err
	}

	leftPointer, _, err := pointerOf(left)

	if err != nil {
		return err
	}

	rightPointer, _, err := pointerOf(right)

	if err != nil {
		return err
	}

	outputPointer, _, err := pointerOf(output)

	if err != nil {
		return err
	}

	return copyConcatBytes(
		unsafe.Slice((*byte)(outputPointer), output.Bytes()),
		unsafe.Slice((*byte)(leftPointer), left.Bytes()),
		unsafe.Slice((*byte)(rightPointer), right.Bytes()),
		left.Shape().Dims(),
		right.Shape().Dims(),
		axis,
		elementSize,
	)
}

func copyConcatBytes(
	output []byte,
	left []byte,
	right []byte,
	leftDimensions []int,
	rightDimensions []int,
	axis int,
	elementSize int,
) error {
	inner := productInts(leftDimensions[axis+1:])
	outer := productInts(leftDimensions[:axis])
	leftBlockBytes := leftDimensions[axis] * inner * elementSize
	rightBlockBytes := rightDimensions[axis] * inner * elementSize
	outputBlockBytes := leftBlockBytes + rightBlockBytes

	if len(output) != outer*outputBlockBytes {
		return fmt.Errorf("shape.concat output byte length mismatch")
	}

	for outerIndex := range outer {
		outputBase := outerIndex * outputBlockBytes
		leftBase := outerIndex * leftBlockBytes
		rightBase := outerIndex * rightBlockBytes

		copy(output[outputBase:outputBase+leftBlockBytes], left[leftBase:leftBase+leftBlockBytes])
		copy(
			output[outputBase+leftBlockBytes:outputBase+outputBlockBytes],
			right[rightBase:rightBase+rightBlockBytes],
		)
	}

	return nil
}

func runConcatDeviceIntrinsic(
	resolver *bindResolver,
	output tensor.Tensor,
	left tensor.Tensor,
	right tensor.Tensor,
	axis int,
) error {
	deviceBackend, ok := resolver.dispatcher.deviceBackend.(concatDevice)

	if !ok {
		return fmt.Errorf(
			"shape.concat: backend %T cannot run %s tensor",
			resolver.dispatcher.deviceBackend,
			left.Location(),
		)
	}

	leftPointer, _, err := pointerOf(left)

	if err != nil {
		return err
	}

	rightPointer, _, err := pointerOf(right)

	if err != nil {
		return err
	}

	outputPointer, _, err := pointerOf(output)

	if err != nil {
		return err
	}

	leftDimensions := left.Shape().Dims()
	outer := productInts(leftDimensions[:axis])

	if axis == len(leftDimensions)-1 {
		return runConcatLastDimDevice(deviceBackend, leftPointer, rightPointer, outputPointer, output, left, right)
	}

	if axis == 0 || outer == 1 {
		deviceBackend.Concat(leftPointer, rightPointer, outputPointer, left.Bytes(), right.Bytes(), left.DType())
		return nil
	}

	return fmt.Errorf(
		"shape.concat dim %d with outer count %d needs a strided device concat",
		axis, outer,
	)
}

func runConcatLastDimDevice(
	deviceBackend concatDevice,
	leftPointer unsafe.Pointer,
	rightPointer unsafe.Pointer,
	outputPointer unsafe.Pointer,
	output tensor.Tensor,
	left tensor.Tensor,
	right tensor.Tensor,
) error {
	elementSize, err := left.DType().Size()

	if err != nil {
		return err
	}

	leftDimensions := left.Shape().Dims()
	rightDimensions := right.Shape().Dims()
	lastAxis := len(leftDimensions) - 1
	leftRowBytes := leftDimensions[lastAxis] * elementSize
	rightRowBytes := rightDimensions[lastAxis] * elementSize
	rowBytes := leftRowBytes + rightRowBytes

	deviceBackend.ConcatLastDim(
		leftPointer,
		rightPointer,
		outputPointer,
		leftRowBytes,
		rightRowBytes,
		rowBytes,
		output.Bytes(),
		left.DType(),
	)

	return nil
}

func runLastTokenDeviceIntrinsic(
	resolver *bindResolver,
	input tensor.Tensor,
	batch int,
	sequence int,
	rowElements int,
) (tensor.Tensor, error) {
	deviceBackend, ok := resolver.dispatcher.deviceBackend.(lastTokenDevice)

	if !ok {
		return nil, fmt.Errorf(
			"shape.last_token: backend %T cannot run %s tensor",
			resolver.dispatcher.deviceBackend,
			input.Location(),
		)
	}

	output, err := resolver.allocateOutput()

	if err != nil {
		return nil, err
	}

	inputPointer, _, err := pointerOf(input)

	if err != nil {
		return nil, err
	}

	outputPointer, _, err := pointerOf(output)

	if err != nil {
		return nil, err
	}

	elementSize, err := input.DType().Size()

	if err != nil {
		return nil, err
	}

	rowBytes := rowElements * elementSize
	outBytes := batch * rowBytes

	deviceBackend.LastToken(
		inputPointer,
		outputPointer,
		sequence,
		rowBytes,
		outBytes,
		input.DType(),
	)

	return output, nil
}

func copyTensorStorage(destination tensor.Tensor, source tensor.Tensor, elements int) error {
	if destination.DType() != source.DType() {
		return fmt.Errorf(
			"copy tensor storage dtype mismatch: destination %s, source %s",
			destination.DType(),
			source.DType(),
		)
	}

	elementSize, err := source.DType().Size()

	if err != nil {
		return err
	}

	byteCount := elements * elementSize
	destinationPointer, _, err := pointerOf(destination)

	if err != nil {
		return err
	}

	sourcePointer, _, err := pointerOf(source)

	if err != nil {
		return err
	}

	copy(
		unsafe.Slice((*byte)(destinationPointer), byteCount),
		unsafe.Slice((*byte)(sourcePointer), byteCount),
	)

	return nil
}
