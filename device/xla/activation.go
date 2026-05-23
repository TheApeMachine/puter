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

/*
NewUnaryActivationLowering constructs a unary activation lowering descriptor.
*/
func NewUnaryActivationLowering(operationName string) UnaryActivationLowering {
	return UnaryActivationLowering{operationName: operationName}
}

/*
Name returns the XLA operation identifier.
*/
func (unaryActivationLowering UnaryActivationLowering) Name() string {
	return unaryActivationLowering.operationName
}

/*
ProgramKey builds the compile cache key for a unary activation launch.
*/
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

/*
RegisterStandardActivations registers unary activation lowerings on the registry.
*/
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
