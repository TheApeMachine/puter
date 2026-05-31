package execution

import (
	"fmt"
	"unsafe"

	"github.com/theapemachine/manifesto/tensor"
)

func isResonantMultiOutputMethod(method string) bool {
	switch method {
	case "ResonantUpdateForward", "ResonantUpdateBackward":
		return true
	default:
		return false
	}
}

func (resolver *bindResolver) allocateResonantOutputs() error {
	if len(resolver.bind.OutputNames) == 0 {
		return fmt.Errorf("bind op %q: resonant outputs are required", resolver.node.Op)
	}

	shapeTensor, err := resolver.resolveInputTensor(resolver.node.Inputs[0])

	if err != nil {
		return fmt.Errorf("bind op %q: resonant shape input: %w", resolver.node.Op, err)
	}

	outputShape := shapeTensor.Shape()
	outputDType := shapeTensor.DType()
	byteCount, err := outputDType.BytesFor(outputShape.Len())

	if err != nil {
		return err
	}

	resolver.outputsByName = make(map[string]tensor.Tensor, len(resolver.bind.OutputNames))

	for _, outputName := range resolver.bind.OutputNames {
		allocated, allocErr := resolver.dispatcher.allocateOutput(
			resolver.node,
			outputShape,
			outputDType,
			byteCount,
		)

		if allocErr != nil {
			return fmt.Errorf("bind op %q: allocate %q: %w", resolver.node.Op, outputName, allocErr)
		}

		resolver.outputsByName[outputName] = allocated
	}

	primaryName := resolver.bind.OutputNames[0]
	resolver.output = resolver.outputsByName[primaryName]
	resolver.outputShape = outputShape
	resolver.outputDType = outputDType

	return nil
}

func (resolver *bindResolver) storeResonantOutputs() {
	for outputName, outputTensor := range resolver.outputsByName {
		resolver.dispatcher.values.set(outputName, outputTensor)
	}

	if len(resolver.bind.OutputNames) > 0 {
		resolver.storeOutput(resolver.outputsByName[resolver.bind.OutputNames[0]])
	}
}

func callResonantUpdateForward(
	deviceBackend executionDevice,
	configFields map[string]any,
	args []any,
) error {
	if len(args) != 14 {
		return fmt.Errorf("router: ResonantUpdateForward expects 14 args, got %d", len(args))
	}

	config, err := castResonantUpdateConfig(configFields)

	if err != nil {
		return err
	}

	x, y, vr, vi, diag, xOut, yOut, aOut, bOut, invROut, err := castTenPointers(
		args[:10],
		"ResonantUpdateForward",
	)

	if err != nil {
		return err
	}

	batchTime, headCount, headDim, err := castThreeInts(args[10:13], "ResonantUpdateForward")

	if err != nil {
		return err
	}

	format, err := castDType(args[13], "ResonantUpdateForward", "format")

	if err != nil {
		return err
	}

	deviceBackend.ResonantUpdateForward(
		x, y, vr, vi, diag,
		xOut, yOut, aOut, bOut, invROut,
		batchTime, headCount, headDim,
		config,
		format,
	)

	return nil
}

func callResonantUpdateBackward(
	deviceBackend executionDevice,
	configFields map[string]any,
	args []any,
) error {
	if len(args) != 16 {
		return fmt.Errorf("router: ResonantUpdateBackward expects 16 args, got %d", len(args))
	}

	config, err := castResonantUpdateConfig(configFields)

	if err != nil {
		return err
	}

	gradXOut, gradYOut, x, y, diag, a, b, invR, gradX, gradY, gradVR, gradVI, err := castTwelvePointers(
		args[:12],
		"ResonantUpdateBackward",
	)

	if err != nil {
		return err
	}

	batchTime, headCount, headDim, err := castThreeInts(args[12:15], "ResonantUpdateBackward")

	if err != nil {
		return err
	}

	format, err := castDType(args[15], "ResonantUpdateBackward", "format")

	if err != nil {
		return err
	}

	deviceBackend.ResonantUpdateBackward(
		gradXOut, gradYOut,
		x, y, diag, a, b, invR,
		gradX, gradY, gradVR, gradVI,
		batchTime, headCount, headDim,
		config,
		format,
	)

	return nil
}

func castTenPointers(args []any, method string) (
	first, second, third, fourth, fifth,
	sixth, seventh, eighth, ninth, tenth unsafe.Pointer,
	err error,
) {
	if len(args) != 10 {
		return nil, nil, nil, nil, nil, nil, nil, nil, nil, nil,
			fmt.Errorf("router: %s expects 10 pointers, got %d", method, len(args))
	}

	pointers := make([]unsafe.Pointer, 10)

	for index, raw := range args {
		pointer, castErr := castPointer(raw, method, fmt.Sprintf("arg%d", index))

		if castErr != nil {
			return nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, castErr
		}

		pointers[index] = pointer
	}

	return pointers[0], pointers[1], pointers[2], pointers[3], pointers[4],
		pointers[5], pointers[6], pointers[7], pointers[8], pointers[9], nil
}

func castTwelvePointers(args []any, method string) (
	first, second, third, fourth, fifth, sixth,
	seventh, eighth, ninth, tenth, eleventh, twelfth unsafe.Pointer,
	err error,
) {
	if len(args) != 12 {
		return nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil,
			fmt.Errorf("router: %s expects 12 pointers, got %d", method, len(args))
	}

	pointers := make([]unsafe.Pointer, 12)

	for index, raw := range args {
		pointer, castErr := castPointer(raw, method, fmt.Sprintf("arg%d", index))

		if castErr != nil {
			return nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, castErr
		}

		pointers[index] = pointer
	}

	return pointers[0], pointers[1], pointers[2], pointers[3], pointers[4],
		pointers[5], pointers[6], pointers[7], pointers[8], pointers[9],
		pointers[10], pointers[11], nil
}
