package xla

import (
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func RegisterRopeLowerings(registry *LoweringRegistry) {
	registry.Register(NewVariadicLowering("rope_pairs", 3))
	registry.Register(RoPELowering{})
}

type RoPELowering struct{}

func (ropeLowering RoPELowering) Name() string {
	return "rope"
}

func (ropeLowering RoPELowering) ProgramKey(
	context LoweringContext,
	floatParams []float64,
	intParams []int64,
) (ProgramKey, error) {
	if len(context.InputDTypes) != 1 || len(context.InputShapes) != 1 {
		return ProgramKey{}, &loweringError{message: "rope requires one input tensor"}
	}

	if context.InputShapes[0].Rank() != 3 {
		return ProgramKey{}, &loweringError{message: "rope requires rank-3 input"}
	}

	return ProgramKey{
		Operation:   "rope",
		DTypes:      []dtype.DType{context.InputDTypes[0], context.OutputDType},
		Shapes:      []tensor.Shape{context.InputShapes[0], context.OutputShape},
		FloatParams: floatParams,
		IntParams:   intParams,
	}, nil
}

func RegisterAttentionLowerings(registry *LoweringRegistry) {
	registry.Register(NewVariadicLowering("scaled_dot_product_attention", 3))
	registry.Register(NewVariadicLowering("multi_head_attention", 3))
}
