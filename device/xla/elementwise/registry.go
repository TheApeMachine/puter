package elementwise

import (
	"fmt"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device/xla"
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

func (unaryElementwiseLowering UnaryElementwiseLowering) Name() string {
	return unaryElementwiseLowering.operationName
}

func (unaryElementwiseLowering UnaryElementwiseLowering) ProgramKey(
	context xla.LoweringContext,
	floatParams []float64,
	intParams []int64,
) (xla.ProgramKey, error) {
	if err := xla.ValidateShape(context.OutputShape); err != nil {
		return xla.ProgramKey{}, err
	}

	if len(context.InputDTypes) != 1 || len(context.InputShapes) != 1 {
		return xla.ProgramKey{}, fmt.Errorf("unary elementwise requires one input tensor")
	}

	return xla.ProgramKey{
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

func NewBinaryElementwiseLowering(operationName string) BinaryElementwiseLowering {
	return BinaryElementwiseLowering{operationName: operationName}
}

func (binaryElementwiseLowering BinaryElementwiseLowering) Name() string {
	return binaryElementwiseLowering.operationName
}

func (binaryElementwiseLowering BinaryElementwiseLowering) ProgramKey(
	context xla.LoweringContext,
	floatParams []float64,
	intParams []int64,
) (xla.ProgramKey, error) {
	if err := xla.ValidateShape(context.OutputShape); err != nil {
		return xla.ProgramKey{}, err
	}

	if len(context.InputDTypes) != 2 || len(context.InputShapes) != 2 {
		return xla.ProgramKey{}, fmt.Errorf("binary elementwise requires two input tensors")
	}

	broadcastShape, err := xla.BroadcastShape(context.InputShapes[0], context.InputShapes[1])

	if err != nil {
		return xla.ProgramKey{}, err
	}

	return xla.ProgramKey{
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
RegisterLowerings registers unary and binary elementwise lowerings.
*/
func RegisterLowerings(registry *xla.LoweringRegistry) {
	unaryOperations := []string{"abs", "neg", "sqrt", "relu"}
	for _, operationName := range unaryOperations {
		registry.Register(NewUnaryElementwiseLowering(operationName))
	}

	binaryOperations := []string{"add", "sub", "mul", "div", "max", "min"}
	for _, operationName := range binaryOperations {
		registry.Register(NewBinaryElementwiseLowering(operationName))
	}
}
