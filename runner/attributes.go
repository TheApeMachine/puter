package runner

import (
	"encoding/binary"
	"fmt"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
)

func appendKernelAttributes(
	memory tensor.Backend,
	node *ir.Node,
	kernel string,
	args []tensor.Tensor,
) ([]tensor.Tensor, error) {
	if len(args) == 0 {
		return nil, fmt.Errorf("runner: node %q has no dispatch arguments", node.ID())
	}

	switch kernel {
	case "view_as_heads":
		return appendInt32AttributeBeforeOutput(memory, node, "num_heads", args)
	default:
		return args, nil
	}
}

func appendInt32AttributeBeforeOutput(
	memory tensor.Backend,
	node *ir.Node,
	attributeName string,
	args []tensor.Tensor,
) ([]tensor.Tensor, error) {
	value, err := nodeIntAttribute(node, attributeName)

	if err != nil {
		return nil, fmt.Errorf("runner: node %q: %w", node.ID(), err)
	}

	scalar, err := uploadInt32Scalar(memory, value)

	if err != nil {
		return nil, err
	}

	output := args[len(args)-1]
	inputs := args[:len(args)-1]

	return append(append(inputs, scalar), output), nil
}

func uploadInt32Scalar(memory tensor.Backend, value int) (tensor.Tensor, error) {
	shape, err := tensor.NewShape([]int{1})

	if err != nil {
		return nil, err
	}

	buffer := make([]byte, 4)
	binary.LittleEndian.PutUint32(buffer, uint32(value))

	return memory.Upload(shape, dtype.Int32, buffer)
}
