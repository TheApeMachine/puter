package xla

import (
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
UnaryActivationLowering describes an XLA unary activation compile key.
*/
type UnaryActivationLowering struct {
	operationName string
}

func NewUnaryActivationLowering(operationName string) UnaryActivationLowering {
	return UnaryActivationLowering{operationName: operationName}
}

func (unaryActivationLowering UnaryActivationLowering) Name() string {
	return unaryActivationLowering.operationName
}

func (unaryActivationLowering UnaryActivationLowering) ProgramKey(
	context LoweringContext,
	floatParams []float64,
	intParams []int64,
) (ProgramKey, error) {
	if err := ValidateShape(context.OutputShape); err != nil {
		return ProgramKey{}, err
	}

	if len(context.InputDTypes) != 1 || len(context.InputShapes) != 1 {
		return ProgramKey{}, &loweringError{message: "unary activation requires one input tensor"}
	}

	return ProgramKey{
		Operation:   unaryActivationLowering.operationName,
		DTypes:      []dtype.DType{context.InputDTypes[0], context.OutputDType},
		Shapes:      []tensor.Shape{context.InputShapes[0], context.OutputShape},
		FloatParams: floatParams,
		IntParams:   intParams,
	}, nil
}

func RegisterStandardActivations(registry *LoweringRegistry) {
	operations := []string{
		"relu", "exp", "log", "log1p", "expm1", "sigmoid", "log_sigmoid",
		"tanh", "silu", "swish", "gelu_tanh", "gelu", "leaky_relu", "elu",
		"celu", "selu", "softplus", "mish", "softsign", "hard_sigmoid",
		"hard_swish", "hard_tanh", "hard_gelu", "quick_gelu", "tanh_shrink",
	}

	for _, operationName := range operations {
		registry.Register(NewUnaryActivationLowering(operationName))
	}
}

/*
UnaryElementwiseLowering describes an XLA unary elementwise compile key.
*/
type UnaryElementwiseLowering struct {
	operationName string
}

func NewUnaryElementwiseLowering(operationName string) UnaryElementwiseLowering {
	return UnaryElementwiseLowering{operationName: operationName}
}

func (unaryElementwiseLowering UnaryElementwiseLowering) Name() string {
	return unaryElementwiseLowering.operationName
}

func (unaryElementwiseLowering UnaryElementwiseLowering) ProgramKey(
	context LoweringContext,
	floatParams []float64,
	intParams []int64,
) (ProgramKey, error) {
	if err := ValidateShape(context.OutputShape); err != nil {
		return ProgramKey{}, err
	}

	if len(context.InputDTypes) != 1 || len(context.InputShapes) != 1 {
		return ProgramKey{}, &loweringError{message: "unary elementwise requires one input tensor"}
	}

	return ProgramKey{
		Operation:   unaryElementwiseLowering.operationName,
		DTypes:      []dtype.DType{context.InputDTypes[0], context.OutputDType},
		Shapes:      []tensor.Shape{context.InputShapes[0], context.OutputShape},
		FloatParams: floatParams,
		IntParams:   intParams,
	}, nil
}

type BinaryElementwiseLowering struct {
	operationName string
}

func NewBinaryElementwiseLowering(operationName string) BinaryElementwiseLowering {
	return BinaryElementwiseLowering{operationName: operationName}
}

func (binaryElementwiseLowering BinaryElementwiseLowering) Name() string {
	return binaryElementwiseLowering.operationName
}

func (binaryElementwiseLowering BinaryElementwiseLowering) ProgramKey(
	context LoweringContext,
	floatParams []float64,
	intParams []int64,
) (ProgramKey, error) {
	if err := ValidateShape(context.OutputShape); err != nil {
		return ProgramKey{}, err
	}

	if len(context.InputDTypes) != 2 || len(context.InputShapes) != 2 {
		return ProgramKey{}, &loweringError{message: "binary elementwise requires two input tensors"}
	}

	broadcastShape, err := BroadcastShape(context.InputShapes[0], context.InputShapes[1])

	if err != nil {
		return ProgramKey{}, err
	}

	return ProgramKey{
		Operation: binaryElementwiseLowering.operationName,
		DTypes: []dtype.DType{
			context.InputDTypes[0],
			context.InputDTypes[1],
			context.OutputDType,
		},
		Shapes: []tensor.Shape{
			context.InputShapes[0],
			context.InputShapes[1],
			broadcastShape,
			context.OutputShape,
		},
		FloatParams: floatParams,
		IntParams:   intParams,
	}, nil
}

func RegisterElementwiseLowerings(registry *LoweringRegistry) {
	unaryOperations := []string{"abs", "neg", "sqrt", "relu"}
	for _, operationName := range unaryOperations {
		registry.Register(NewUnaryElementwiseLowering(operationName))
	}

	binaryOperations := []string{"add", "sub", "mul", "div", "max", "min"}
	for _, operationName := range binaryOperations {
		registry.Register(NewBinaryElementwiseLowering(operationName))
	}
}
