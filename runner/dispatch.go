package runner

import (
	"fmt"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device/cpu/elementwise"
	"github.com/theapemachine/puter/device/cpu/matmul"
	"github.com/theapemachine/puter/device/cpu/shape"
	"github.com/theapemachine/puter/kernels"
)

func dispatchNode(
	node *ir.Node,
	location tensor.Location,
	memory tensor.Backend,
	checkpointPath string,
	weights *weightCache,
	bindings *manifestBindings,
	tensorWorkspace *workspace,
) error {
	kernel := kernelName(node.OperationID())

	if kernel == "" {
		return fmt.Errorf("runner: unsupported operation %q", node.OperationID())
	}

	storageDType := nodeStorageDType(
		node,
		tensorWorkspace,
		checkpointPath,
		weights,
		bindings,
	)

	outputShape, err := outputShapeForNode(
		node,
		kernel,
		tensorWorkspace,
		checkpointPath,
		weights,
		bindings,
	)

	if err != nil {
		return err
	}

	output, err := allocateTensor(memory, outputShape, storageDType)

	if err != nil {
		return err
	}

	args, err := nodeArguments(
		node,
		memory,
		checkpointPath,
		weights,
		bindings,
		tensorWorkspace,
		output,
		storageDType,
	)

	if err != nil {
		_ = output.Close()
		return err
	}

	args = orderKernelArguments(kernel, args)

	if location == tensor.Host {
		err = dispatchHostKernel(kernel, args)
	} else if kernel == "rope" {
		err = dispatchRoPE(node, args)
	} else if kernel == "rmsnorm" {
		err = dispatchRMSNorm(node, args)
	} else {
		err = dispatchRegisteredKernel(kernel, location, args)
	}

	if err != nil {
		_ = output.Close()
		return err
	}

	tensorWorkspace.Store(node.ID(), output)

	return nil
}

func nodeArguments(
	node *ir.Node,
	memory tensor.Backend,
	checkpointPath string,
	weights *weightCache,
	bindings *manifestBindings,
	tensorWorkspace *workspace,
	output tensor.Tensor,
	storageDType dtype.DType,
) ([]tensor.Tensor, error) {
	args := make([]tensor.Tensor, 0, len(node.Inputs())+4)

	for _, inputNode := range node.Inputs() {
		value, ok := tensorWorkspace.Load(inputNode.ID())

		if !ok {
			return nil, fmt.Errorf("missing input %q", inputNode.ID())
		}

		args = append(args, value)
	}

	weightName := bindings.weightTensorName(node.ID())

	if weightName == "" {
		weightName = weightTensorName(node)
	}

	if weightName != "" {
		weight, err := weights.Tensor(checkpointPath, weightName)

		if err != nil {
			return nil, err
		}

		args = append(args, weight)
	}

	if kernelUsesBias(kernelName(node.OperationID())) {
		bias, err := resolveBiasTensor(
			node,
			memory,
			checkpointPath,
			weights,
			bindings,
			storageDType,
		)

		if err != nil {
			return nil, err
		}

		args = append(args, bias)
	}

	args = append(args, output)

	args, attributeErr := appendKernelAttributes(memory, node, kernelName(node.OperationID()), args)

	if attributeErr != nil {
		return nil, attributeErr
	}

	return args, nil
}

func kernelUsesBias(kernel string) bool {
	return kernel == "linear"
}

func resolveBiasTensor(
	node *ir.Node,
	memory tensor.Backend,
	checkpointPath string,
	weights *weightCache,
	bindings *manifestBindings,
	storageDType dtype.DType,
) (tensor.Tensor, error) {
	biasName := bindings.biasTensorName(node.ID())

	if biasName != "" {
		return weights.Tensor(checkpointPath, biasName)
	}

	return zeroBiasTensor(memory, node, storageDType)
}

func zeroBiasTensor(
	memory tensor.Backend,
	node *ir.Node,
	storageDType dtype.DType,
) (tensor.Tensor, error) {
	outputDims := node.Shape().Dims()

	if len(outputDims) == 0 {
		return nil, fmt.Errorf("runner: linear node %q has empty output shape", node.ID())
	}

	featureCount := outputDims[len(outputDims)-1]
	biasShape, err := tensor.NewShape([]int{featureCount})

	if err != nil {
		return nil, err
	}

	return zeroTensor(memory, biasShape, storageDType)
}

func dispatchRegisteredKernel(
	kernel string,
	location tensor.Location,
	args []tensor.Tensor,
) error {
	signature := kernelSignature(args)
	registered, ok := kernels.Default.LookupLocation(kernel, signature, location)

	if !ok {
		return fmt.Errorf("kernel %q not registered for %s", kernel, location)
	}

	return registered.Run(args...)
}

func kernelSignature(args []tensor.Tensor) kernels.Signature {
	inputTypes := make([]dtype.DType, 0, len(args)-1)

	for index := 0; index < len(args)-1; index++ {
		inputTypes = append(inputTypes, args[index].DType())
	}

	outputDType := args[len(args)-1].DType()

	return kernels.Signature{
		Layout:  tensor.LayoutDense,
		Inputs:  inputTypes,
		Outputs: []dtype.DType{outputDType},
	}
}

func nodeStorageDType(
	node *ir.Node,
	tensorWorkspace *workspace,
	checkpointPath string,
	weights *weightCache,
	bindings *manifestBindings,
) dtype.DType {
	if storageDType := weightStorageDType(node, checkpointPath, weights, bindings); storageDType != dtype.Invalid {
		return storageDType
	}

	for _, inputNode := range node.Inputs() {
		value, ok := tensorWorkspace.Load(inputNode.ID())

		if !ok || inputNode.OpType() == ir.OpInput {
			continue
		}

		if value.DType().IsFloat() {
			return value.DType()
		}
	}

	if weightDType, ok := node.Metadata()["weight_dtype"].(string); ok {
		parsed, err := dtype.Parse(weightDType)

		if err == nil {
			return parsed
		}
	}

	storageDType := node.ValueType().DType

	if storageDType == dtype.Invalid || storageDType == dtype.Float64 {
		return dtype.Float32
	}

	return storageDType
}

func weightStorageDType(
	node *ir.Node,
	checkpointPath string,
	weights *weightCache,
	bindings *manifestBindings,
) dtype.DType {
	weightName := bindings.weightTensorName(node.ID())

	if weightName == "" {
		weightName = weightTensorName(node)
	}

	if weightName == "" || checkpointPath == "" || weights == nil {
		return dtype.Invalid
	}

	weight, err := weights.Tensor(checkpointPath, weightName)

	if err != nil {
		return dtype.Invalid
	}

	return weight.DType()
}

func dispatchHostKernel(kernel string, args []tensor.Tensor) error {
	switch kernel {
	case "matmul":
		return matmul.RunMatMulFloat32(args...)
	case "relu":
		return elementwise.RunRelu(args...)
	case "add":
		return elementwise.RunAdd(args...)
	case "mul":
		return elementwise.RunMul(args...)
	case "page_write":
		return shape.RunPageWrite(args...)
	case "page_gather":
		return shape.RunPageGather(args...)
	default:
		return fmt.Errorf("host kernel %q is not implemented", kernel)
	}
}

func allocateTensor(
	memory tensor.Backend,
	shape tensor.Shape,
	storageDType dtype.DType,
) (tensor.Tensor, error) {
	return zeroTensor(memory, shape, storageDType)
}
