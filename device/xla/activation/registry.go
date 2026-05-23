package activation

import (
	"fmt"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device/xla"
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
	context xla.LoweringContext,
	floatParams []float64,
	intParams []int64,
) (xla.ProgramKey, error) {
	if err := xla.ValidateShape(context.OutputShape); err != nil {
		return xla.ProgramKey{}, err
	}

	if len(context.InputDTypes) != 1 || len(context.InputShapes) != 1 {
		return xla.ProgramKey{}, fmt.Errorf("unary activation requires one input tensor")
	}

	return xla.ProgramKey{
		Operation:   unaryActivationLowering.operationName,
		DTypes:      []dtype.DType{context.InputDTypes[0], context.OutputDType},
		Shapes:      []tensor.Shape{context.InputShapes[0], context.OutputShape},
		FloatParams: floatParams,
		IntParams:   intParams,
	}, nil
}

/*
RegisterLowerings registers unary activation lowerings on the registry.
*/
func RegisterLowerings(registry *xla.LoweringRegistry) {
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
