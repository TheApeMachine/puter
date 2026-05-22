package runner

import (
	"fmt"
	"strings"

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
	} else if kernel == "modulated_layernorm" {
		err = dispatchModulatedLayerNorm(node, args)
	} else if kernel == "gated_residual" {
		err = dispatchGatedResidual(node, args)
	} else if kernel == "multi_axis_rope" {
		err = dispatchMultiAxisRoPE(node, args)
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
		weight, err := weights.TensorForNode(weightFilePath(node, checkpointPath), weightName, node)

		if err != nil {
			return nil, err
		}

		args = append(args, weight)
	}

	if kernelUsesBias(kernelName(node.OperationID())) {
		bias, err := resolveBiasTensor(
			node,
			memory,
			weightFilePath(node, checkpointPath),
			weights,
			bindings,
			storageDType,
		)

		if err != nil {
			return nil, err
		}

		args = append(args, bias)
	}

	if kernelName(node.OperationID()) == "batchnorm_denorm" {
		normArgs, err := resolveBatchNormDenormTensors(
			node,
			weightFilePath(node, checkpointPath),
			weights,
		)

		if err != nil {
			return nil, err
		}

		args = append(args, normArgs...)
	}

	args = append(args, output)

	args, attributeErr := appendKernelAttributes(memory, node, kernelName(node.OperationID()), args)

	if attributeErr != nil {
		return nil, attributeErr
	}

	return args, nil
}

func resolveBatchNormDenormTensors(
	node *ir.Node,
	checkpointPath string,
	weights *weightCache,
) ([]tensor.Tensor, error) {
	if weights == nil {
		return nil, fmt.Errorf("runner: batchnorm_denorm node %q requires weights", node.ID())
	}

	meanName, ok := nodeOptionalStringAttribute(node, "mean")
	if !ok || meanName == "" {
		return nil, fmt.Errorf("runner: batchnorm_denorm node %q missing mean", node.ID())
	}

	varianceName, ok := nodeOptionalStringAttribute(node, "variance")
	if !ok || varianceName == "" {
		return nil, fmt.Errorf("runner: batchnorm_denorm node %q missing variance", node.ID())
	}

	mean, err := weights.Tensor(checkpointPath, meanName)
	if err != nil {
		return nil, err
	}

	variance, err := weights.Tensor(checkpointPath, varianceName)
	if err != nil {
		return nil, err
	}

	return []tensor.Tensor{mean, variance}, nil
}

func kernelUsesBias(kernel string) bool {
	switch kernel {
	case "linear", "conv1d", "conv2d", "conv3d", "conv_transpose2d",
		"layernorm", "groupnorm", "instancenorm", "batchnorm_eval":
		return true
	default:
		return false
	}
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

	if weightName := weightTensorName(node); strings.HasSuffix(weightName, ".weight") && weights != nil {
		inferredBiasName := strings.TrimSuffix(weightName, ".weight") + ".bias"
		bias, err := weights.Tensor(checkpointPath, inferredBiasName)

		if err == nil {
			return bias, nil
		}
	}

	return zeroBiasTensor(memory, node, storageDType)
}

func zeroBiasTensor(
	memory tensor.Backend,
	node *ir.Node,
	storageDType dtype.DType,
) (tensor.Tensor, error) {
	featureCount, err := nodeIntAttribute(node, "out_channels")

	if err != nil {
		outputDims := node.Shape().Dims()

		if len(outputDims) == 0 {
			return nil, fmt.Errorf("runner: bias node %q has empty output shape", node.ID())
		}

		featureCount = outputDims[len(outputDims)-1]
	}

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
		return fmt.Errorf("kernel %q signature inputs=%v outputs=%v not registered for %s",
			kernel,
			signature.Inputs,
			signature.Outputs,
			location,
		)
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

	if node.OperationID() == ir.OpID("embedding.timestep") {
		return normalizedNodeValueDType(node)
	}

	for _, inputNode := range node.Inputs() {
		value, ok := tensorWorkspace.Load(inputNode.ID())

		if !ok {
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

	return normalizedNodeValueDType(node)
}

func normalizedNodeValueDType(node *ir.Node) dtype.DType {
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
	if weightDType, ok := node.Metadata()["weight_dtype"].(string); ok && weightDType != "" {
		parsed, err := dtype.Parse(weightDType)

		if err == nil {
			return parsed
		}
	}

	weightName := bindings.weightTensorName(node.ID())

	if weightName == "" {
		weightName = weightTensorName(node)
	}

	weightPath := weightFilePath(node, checkpointPath)

	if weightName == "" || weightPath == "" || weights == nil {
		return dtype.Invalid
	}

	weight, err := weights.TensorForNode(weightPath, weightName, node)

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
	if allocator, ok := memory.(uninitializedTensorAllocator); ok {
		return allocator.NewEmpty(shape, storageDType)
	}

	return zeroTensor(memory, shape, storageDType)
}

type uninitializedTensorAllocator interface {
	NewEmpty(shape tensor.Shape, storageDType dtype.DType) (tensor.Tensor, error)
}
