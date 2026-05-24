package xla

import (
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device/xla/internal/hlo"
)

type PairLossLowering struct {
	lossKind string
}

func NewPairLossLowering(lossKind string) PairLossLowering {
	return PairLossLowering{lossKind: lossKind}
}

func (pairLossLowering PairLossLowering) Name() string {
	return hlo.PairLossOperationName(pairLossLowering.lossKind)
}

func (pairLossLowering PairLossLowering) ProgramKey(
	context LoweringContext,
	floatParams []float64,
	intParams []int64,
) (ProgramKey, error) {
	_ = floatParams
	_ = intParams

	if context.OutputShape.Rank() != 0 {
		return ProgramKey{}, &loweringError{message: "pair loss output must be scalar"}
	}

	if len(context.InputDTypes) != 2 || len(context.InputShapes) != 2 {
		return ProgramKey{}, &loweringError{message: "pair loss requires two input tensors"}
	}

	scalarShape, err := tensor.NewShape([]int{})

	if err != nil {
		return ProgramKey{}, err
	}

	return ProgramKey{
		Operation: hlo.PairLossOperationName(pairLossLowering.lossKind),
		DTypes: []dtype.DType{
			context.InputDTypes[0],
			context.InputDTypes[1],
			context.OutputDType,
		},
		Shapes: []tensor.Shape{
			context.InputShapes[0],
			context.InputShapes[1],
			scalarShape,
		},
	}, nil
}

func RegisterLossLowerings(registry *LoweringRegistry) {
	lossKinds := []string{"mse", "mae", "huber", "bce", "kl"}

	for _, lossKind := range lossKinds {
		registry.Register(NewPairLossLowering(lossKind))
	}

	registry.Register(CrossEntropyLowering{})
}

type CrossEntropyLowering struct{}

func (crossEntropyLowering CrossEntropyLowering) Name() string {
	return "cross_entropy"
}

func (crossEntropyLowering CrossEntropyLowering) ProgramKey(
	context LoweringContext,
	floatParams []float64,
	intParams []int64,
) (ProgramKey, error) {
	_ = floatParams
	_ = intParams

	if context.OutputShape.Rank() != 0 {
		return ProgramKey{}, &loweringError{message: "cross entropy output must be scalar"}
	}

	if len(context.InputDTypes) != 2 || len(context.InputShapes) != 2 {
		return ProgramKey{}, &loweringError{message: "cross entropy requires logits and targets"}
	}

	if context.InputDTypes[1] != dtype.Int32 {
		return ProgramKey{}, &loweringError{message: "cross entropy targets must be int32"}
	}

	scalarShape, err := tensor.NewShape([]int{})

	if err != nil {
		return ProgramKey{}, err
	}

	return ProgramKey{
		Operation: "cross_entropy",
		DTypes: []dtype.DType{
			context.InputDTypes[0],
			context.InputDTypes[1],
			context.OutputDType,
		},
		Shapes: []tensor.Shape{
			context.InputShapes[0],
			context.InputShapes[1],
			scalarShape,
		},
	}, nil
}
