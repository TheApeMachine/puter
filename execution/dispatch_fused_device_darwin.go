//go:build darwin && cgo

package execution

import (
	"fmt"

	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/codegen"
	"github.com/theapemachine/manifesto/tensor"
	putermetal "github.com/theapemachine/puter/device/metal"
	"github.com/theapemachine/puter/device/metal/fusion"
)

type metalFusionBackend interface {
	MetalContextRef() uintptr
	FusionCache() *fusion.Cache
}

func (dispatcher *dispatcher) tryRunFusedOnMetalDevice(
	runner codegen.ElementwiseRunner,
	node *ast.GraphNode,
	inputSlots []int,
	outputSlot int,
) (bool, error) {
	if dispatcher.memory.Location() != tensor.Metal {
		return false, nil
	}

	program, ok := runner.(codegen.MetalFusionProgram)

	if !ok {
		return false, nil
	}

	fusionBackend, ok := dispatcher.deviceBackend.(metalFusionBackend)

	if !ok || fusionBackend.MetalContextRef() == 0 || fusionBackend.FusionCache() == nil {
		return false, nil
	}

	inputBufferRefs := make([]uintptr, 0, len(program.Inputs()))
	var count int

	for inputIndex, inputName := range program.Inputs() {
		inputTensor, err := dispatcher.fusedInputTensor(node, inputName, inputIndex, inputSlots)

		if err != nil {
			return true, err
		}

		inputPointer, elementCount, err := pointerOf(inputTensor)

		if err != nil {
			return true, fmt.Errorf("fused node input %q: %w", inputName, err)
		}

		bufferRef := putermetal.BufferRefFromDispatch(inputPointer)

		if bufferRef == 0 {
			return false, nil
		}

		inputBufferRefs = append(inputBufferRefs, bufferRef)

		if count == 0 {
			count = elementCount
		}
	}

	outputTensor, err := dispatcher.allocateLike(node, nil, count)

	if err != nil {
		return true, err
	}

	outputPointer, outputCount, err := pointerOf(outputTensor)

	if err != nil {
		return true, fmt.Errorf("fused node output allocation: %w", err)
	}

	if outputCount != count {
		return true, fmt.Errorf(
			"fused node output count %d does not match input count %d",
			outputCount, count,
		)
	}

	outputBufferRef := putermetal.BufferRefFromDispatch(outputPointer)

	if outputBufferRef == 0 {
		return false, nil
	}

	fusionProgram, err := fusionBackend.FusionCache().Program(
		program.MSLSource(),
		program.MSLKernelName(),
	)

	if err != nil {
		return true, err
	}

	if err := fusionProgram.Dispatch(
		fusionBackend.MetalContextRef(),
		inputBufferRefs,
		outputBufferRef,
		count,
	); err != nil {
		return true, err
	}

	dispatcher.storeNodeValue(node.ID, outputSlot, outputTensor)

	return true, nil
}
