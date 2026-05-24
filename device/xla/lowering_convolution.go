package xla

import (
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

type ConvLowering struct {
	operationName string
	spatialRank   int
	transpose     bool
}

func NewConvLowering(operationName string, spatialRank int, transpose bool) ConvLowering {
	return ConvLowering{
		operationName: operationName,
		spatialRank:   spatialRank,
		transpose:     transpose,
	}
}

func (convLowering ConvLowering) Name() string {
	return convLowering.operationName
}

func (convLowering ConvLowering) ProgramKey(
	context LoweringContext,
	floatParams []float64,
	intParams []int64,
) (ProgramKey, error) {
	_ = floatParams

	if len(context.InputDTypes) != 3 || len(context.InputShapes) != 3 {
		return ProgramKey{}, &loweringError{message: "convolution requires input, weight, and bias tensors"}
	}

	expectedParams := convLowering.spatialRank * 3

	if len(intParams) < expectedParams {
		return ProgramKey{}, &loweringError{message: "convolution int param count mismatch"}
	}

	if context.InputShapes[0].Rank() != convLowering.spatialRank+2 ||
		context.OutputShape.Rank() != convLowering.spatialRank+2 {
		return ProgramKey{}, &loweringError{message: "convolution tensor rank mismatch"}
	}

	shapes := append([]tensor.Shape{}, context.InputShapes...)
	shapes = append(shapes, context.OutputShape)
	dtypes := append([]dtype.DType{}, context.InputDTypes...)
	dtypes = append(dtypes, context.OutputDType)

	return ProgramKey{
		Operation: convLowering.operationName,
		DTypes:    dtypes,
		Shapes:    shapes,
		IntParams: intParams[:expectedParams],
	}, nil
}

func RegisterConvolutionLowerings(registry *LoweringRegistry) {
	registry.Register(NewConvLowering("conv1d", 1, false))
	registry.Register(NewConvLowering("conv2d", 2, false))
	registry.Register(NewConvLowering("conv3d", 3, false))
	registry.Register(NewConvLowering("conv_transpose2d", 2, true))
}
