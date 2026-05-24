package xla

import (
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

type PoolLowering struct {
	operationName string
}

func NewPoolLowering(operationName string) PoolLowering {
	return PoolLowering{operationName: operationName}
}

func (poolLowering PoolLowering) Name() string {
	return poolLowering.operationName
}

func (poolLowering PoolLowering) ProgramKey(
	context LoweringContext,
	floatParams []float64,
	intParams []int64,
) (ProgramKey, error) {
	_ = floatParams

	if len(context.InputDTypes) != 1 || len(context.InputShapes) != 1 {
		return ProgramKey{}, &loweringError{message: "pool requires one input tensor"}
	}

	if context.InputShapes[0].Rank() != 4 || context.OutputShape.Rank() != 4 {
		return ProgramKey{}, &loweringError{message: "pool requires NCHW rank-4 tensors"}
	}

	return ProgramKey{
		Operation: poolLowering.operationName,
		DTypes:    []dtype.DType{context.InputDTypes[0], context.OutputDType},
		Shapes:    []tensor.Shape{context.InputShapes[0], context.OutputShape},
		IntParams: intParams,
	}, nil
}

func RegisterPoolLowerings(registry *LoweringRegistry) {
	poolOperations := []string{
		"max_pool2d", "avg_pool2d", "adaptive_max_pool2d", "adaptive_avg_pool2d",
	}

	for _, operationName := range poolOperations {
		registry.Register(NewPoolLowering(operationName))
	}
}

type VariadicLowering struct {
	operationName string
	inputCount    int
}

func NewVariadicLowering(operationName string, inputCount int) VariadicLowering {
	return VariadicLowering{operationName: operationName, inputCount: inputCount}
}

func (variadicLowering VariadicLowering) Name() string {
	return variadicLowering.operationName
}

func (variadicLowering VariadicLowering) ProgramKey(
	context LoweringContext,
	floatParams []float64,
	intParams []int64,
) (ProgramKey, error) {
	if len(context.InputDTypes) != variadicLowering.inputCount ||
		len(context.InputShapes) != variadicLowering.inputCount {
		return ProgramKey{}, &loweringError{message: "variadic lowering input count mismatch"}
	}

	shapes := append([]tensor.Shape{}, context.InputShapes...)
	shapes = append(shapes, context.OutputShape)
	dtypes := append([]dtype.DType{}, context.InputDTypes...)
	dtypes = append(dtypes, context.OutputDType)

	return ProgramKey{
		Operation:   variadicLowering.operationName,
		DTypes:      dtypes,
		Shapes:      shapes,
		FloatParams: floatParams,
		IntParams:   intParams,
	}, nil
}

func RegisterLayernormLowerings(registry *LoweringRegistry) {
	registry.Register(NewVariadicLowering("layer_norm", 3))
	registry.Register(NewVariadicLowering("rms_norm", 2))
}

func RegisterNormalizationLowerings(registry *LoweringRegistry) {
	registry.Register(NewVariadicLowering("batch_norm_eval", 5))
	registry.Register(NewVariadicLowering("instance_norm", 3))
	registry.Register(NewVariadicLowering("group_norm", 3))
}
