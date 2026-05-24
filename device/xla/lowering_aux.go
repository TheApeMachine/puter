package xla

import (
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func RegisterAuxLowerings(registry *LoweringRegistry) {
	registry.Register(NewVariadicLowering("alibi_bias", 2))
	registry.Register(NullaryLowering{operationName: "causal_mask"})
	registry.Register(UnaryParamLowering{operationName: "dropout"})
	registry.Register(NewVariadicLowering("embedding_lookup", 2))
	registry.Register(NewVariadicLowering("embedding_bag", 3))
	registry.Register(UnaryToScalarLowering{operationName: "greedy_sample"})
}

type NullaryLowering struct {
	operationName string
}

func (nullaryLowering NullaryLowering) Name() string {
	return nullaryLowering.operationName
}

func (nullaryLowering NullaryLowering) ProgramKey(
	context LoweringContext,
	floatParams []float64,
	intParams []int64,
) (ProgramKey, error) {
	_ = floatParams

	if len(context.InputShapes) != 0 {
		return ProgramKey{}, &loweringError{message: "nullary lowering requires no inputs"}
	}

	return ProgramKey{
		Operation: nullaryLowering.operationName,
		DTypes:    []dtype.DType{context.OutputDType},
		Shapes:    []tensor.Shape{context.OutputShape},
		IntParams: intParams,
	}, nil
}

type UnaryParamLowering struct {
	operationName string
}

func (unaryParamLowering UnaryParamLowering) Name() string {
	return unaryParamLowering.operationName
}

func (unaryParamLowering UnaryParamLowering) ProgramKey(
	context LoweringContext,
	floatParams []float64,
	intParams []int64,
) (ProgramKey, error) {
	_ = intParams

	if len(context.InputDTypes) != 1 || len(context.InputShapes) != 1 {
		return ProgramKey{}, &loweringError{message: "unary param lowering requires one input tensor"}
	}

	return ProgramKey{
		Operation:   unaryParamLowering.operationName,
		DTypes:      []dtype.DType{context.InputDTypes[0], context.OutputDType},
		Shapes:      []tensor.Shape{context.InputShapes[0], context.OutputShape},
		FloatParams: floatParams,
		IntParams:   intParams,
	}, nil
}

type UnaryToScalarLowering struct {
	operationName string
}

func (unaryToScalarLowering UnaryToScalarLowering) Name() string {
	return unaryToScalarLowering.operationName
}

func (unaryToScalarLowering UnaryToScalarLowering) ProgramKey(
	context LoweringContext,
	floatParams []float64,
	intParams []int64,
) (ProgramKey, error) {
	_ = floatParams
	_ = intParams

	if context.OutputShape.Rank() != 0 {
		return ProgramKey{}, &loweringError{message: "scalar lowering output must be scalar"}
	}

	if len(context.InputDTypes) != 1 || len(context.InputShapes) != 1 {
		return ProgramKey{}, &loweringError{message: "scalar lowering requires one input tensor"}
	}

	scalarShape, err := tensor.NewShape([]int{})

	if err != nil {
		return ProgramKey{}, err
	}

	return ProgramKey{
		Operation: unaryToScalarLowering.operationName,
		DTypes: []dtype.DType{
			context.InputDTypes[0],
			context.OutputDType,
		},
		Shapes: []tensor.Shape{
			context.InputShapes[0],
			scalarShape,
		},
	}, nil
}
