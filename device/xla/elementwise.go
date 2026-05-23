package xla

import (
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

/*
UnaryElementwiseLowering describes an XLA unary elementwise compile key.
*/
type UnaryElementwiseLowering struct {
	operationName string
}

/*
NewUnaryElementwiseLowering constructs a unary elementwise lowering descriptor.
*/
func NewUnaryElementwiseLowering(operationName string) UnaryElementwiseLowering {
	return UnaryElementwiseLowering{operationName: operationName}
}

/*
Name returns the XLA operation identifier.
*/
func (unaryElementwiseLowering UnaryElementwiseLowering) Name() string {
	return unaryElementwiseLowering.operationName
}

/*
ProgramKey builds the compile cache key for a unary elementwise launch.
*/
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

/*
BinaryElementwiseLowering describes an XLA binary elementwise compile key.
*/
type BinaryElementwiseLowering struct {
	operationName string
}

/*
NewBinaryElementwiseLowering constructs a binary elementwise lowering descriptor.
*/
func NewBinaryElementwiseLowering(operationName string) BinaryElementwiseLowering {
	return BinaryElementwiseLowering{operationName: operationName}
}

/*
Name returns the XLA operation identifier.
*/
func (binaryElementwiseLowering BinaryElementwiseLowering) Name() string {
	return binaryElementwiseLowering.operationName
}

/*
ProgramKey builds the compile cache key for a binary elementwise launch.
*/
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

/*
RegisterElementwiseLowerings registers unary and binary elementwise lowerings.
*/
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

/*
NewDefaultBuilder constructs a builder with standard activation and elementwise lowerings.
*/
func NewDefaultBuilder(target string) *Builder {
	registry := NewLoweringRegistry()
	RegisterStandardActivations(registry)
	RegisterElementwiseLowerings(registry)

	return NewBuilder(NewExecutableCache(), registry, target)
}
