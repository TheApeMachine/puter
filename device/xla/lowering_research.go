package xla

import (
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func RegisterResearchLowerings(registry *LoweringRegistry) {
	registry.Register(ConvertUnaryLowering{operationName: "quant_int8"})
	registry.Register(ConvertUnaryLowering{operationName: "dequant_int8"})
	registry.Register(ConvertUnaryLowering{operationName: "dequant_int4"})
	registry.Register(NewVariadicLowering("belief_update", 2))
	registry.Register(NewVariadicLowering("precision_weight", 2))
	registry.Register(NewVariadicLowering("free_energy", 3))
	registry.Register(NewVariadicLowering("expected_free_energy", 3))
	registry.Register(UnaryParamLowering{operationName: "cyclic_permute"})
	registry.Register(NewVariadicLowering("update_representation", 3))
	registry.Register(NewVariadicLowering("update_weights", 3))
	registry.Register(NewVariadicLowering("hawkes_intensity", 2))
	registry.Register(UnaryParamLowering{operationName: "hawkes_kernel_matrix"})
	registry.Register(UnaryParamLowering{operationName: "hawkes_log_likelihood"})
	registry.Register(NewVariadicLowering("markov_blanket_partition", 2))
}

type ConvertUnaryLowering struct {
	operationName string
}

func (convertLowering ConvertUnaryLowering) Name() string {
	return convertLowering.operationName
}

func (convertLowering ConvertUnaryLowering) ProgramKey(
	context LoweringContext,
	floatParams []float64,
	intParams []int64,
) (ProgramKey, error) {
	if len(context.InputDTypes) != 1 || len(context.InputShapes) != 1 {
		return ProgramKey{}, &loweringError{message: "convert unary requires one input tensor"}
	}

	return ProgramKey{
		Operation:   convertLowering.operationName,
		DTypes:      []dtype.DType{context.InputDTypes[0], context.OutputDType},
		Shapes:      []tensor.Shape{context.InputShapes[0], context.OutputShape},
		FloatParams: floatParams,
		IntParams:   intParams,
	}, nil
}
