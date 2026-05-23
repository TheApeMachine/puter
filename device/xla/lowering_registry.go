package xla

import (
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device/xla/internal/hlo"
)

/*
UnaryActivationLowering describes an XLA unary activation compile key.
*/
type UnaryActivationLowering struct {
	operationName string
}

func NewUnaryActivationLowering(operationName string) UnaryActivationLowering {
	return UnaryActivationLowering{operationName: operationName}
}

func (unaryActivationLowering UnaryActivationLowering) Name() string {
	return unaryActivationLowering.operationName
}

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

func RegisterParametricActivations(registry *LoweringRegistry) {
	unaryParams := []string{
		"prelu_slope", "leaky_relu_slope", "elu_alpha", "celu_alpha",
		"threshold", "snake", "hard_shrink", "soft_shrink",
	}
	dualParams := []string{"hard_tanh_range", "snake_parametric", "rrelu"}
	binaryIndexed := []string{"prelu_v"}
	softmaxOps := []string{"softmax"}

	for _, operationName := range unaryParams {
		registry.Register(NewUnaryActivationLowering(operationName))
	}

	for _, operationName := range dualParams {
		registry.Register(NewUnaryActivationLowering(operationName))
	}

	for _, operationName := range binaryIndexed {
		registry.Register(NewBinaryElementwiseLowering(operationName))
	}

	for _, operationName := range softmaxOps {
		registry.Register(NewUnaryActivationLowering(operationName))
	}
}

func RegisterGatedActivations(registry *LoweringRegistry) {
	gatedOps := []string{
		"glu", "geglu", "geglu_tanh", "swiglu", "reglu", "siglu", "linglu", "seglu",
	}

	for _, operationName := range gatedOps {
		registry.Register(NewBinaryElementwiseLowering(operationName))
	}
}

/*
UnaryElementwiseLowering describes an XLA unary elementwise compile key.
*/
type UnaryElementwiseLowering struct {
	operationName string
}

func NewUnaryElementwiseLowering(operationName string) UnaryElementwiseLowering {
	return UnaryElementwiseLowering{operationName: operationName}
}

func (unaryElementwiseLowering UnaryElementwiseLowering) Name() string {
	return unaryElementwiseLowering.operationName
}

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

func RegisterElementwiseLowerings(registry *LoweringRegistry) {
	unaryOperations := []string{"abs", "neg", "sqrt", "relu"}
	for _, operationName := range unaryOperations {
		registry.Register(NewUnaryElementwiseLowering(operationName))
	}

	binaryOperations := []string{"add", "sub", "mul", "div", "max", "min", "axpy"}
	for _, operationName := range binaryOperations {
		registry.Register(NewBinaryElementwiseLowering(operationName))
	}
}

type ReductionLowering struct {
	operationName string
}

func NewReductionLowering(reductionKind string) ReductionLowering {
	return ReductionLowering{operationName: hlo.ReductionOperationName(reductionKind)}
}

func (reductionLowering ReductionLowering) Name() string {
	return reductionLowering.operationName
}

func (reductionLowering ReductionLowering) ProgramKey(
	context LoweringContext,
	floatParams []float64,
	intParams []int64,
) (ProgramKey, error) {
	_ = floatParams
	_ = intParams

	if err := ValidateShape(context.OutputShape); err != nil {
		return ProgramKey{}, err
	}

	if context.OutputShape.Rank() != 0 {
		return ProgramKey{}, &loweringError{message: "reduction output must be scalar"}
	}

	if len(context.InputDTypes) != 1 || len(context.InputShapes) != 1 {
		return ProgramKey{}, &loweringError{message: "reduction requires one input tensor"}
	}

	scalarShape, err := tensor.NewShape([]int{})

	if err != nil {
		return ProgramKey{}, err
	}

	return ProgramKey{
		Operation: reductionLowering.operationName,
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

func RegisterReductionLowerings(registry *LoweringRegistry) {
	reductionKinds := []string{"sum", "prod", "min", "max", "l1norm"}

	for _, reductionKind := range reductionKinds {
		registry.Register(NewReductionLowering(reductionKind))
	}
}

type DotLowering struct{}

func (dotLowering DotLowering) Name() string {
	return "dot"
}

func (dotLowering DotLowering) ProgramKey(
	context LoweringContext,
	floatParams []float64,
	intParams []int64,
) (ProgramKey, error) {
	_ = floatParams
	_ = intParams

	if context.OutputShape.Rank() != 0 {
		return ProgramKey{}, &loweringError{message: "dot output must be scalar"}
	}

	if len(context.InputDTypes) != 2 || len(context.InputShapes) != 2 {
		return ProgramKey{}, &loweringError{message: "dot requires two input tensors"}
	}

	scalarShape, err := tensor.NewShape([]int{})

	if err != nil {
		return ProgramKey{}, err
	}

	return ProgramKey{
		Operation: "dot",
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

func RegisterDotLowerings(registry *LoweringRegistry) {
	registry.Register(DotLowering{})
}

type MatmulLowering struct{}

func (matmulLowering MatmulLowering) Name() string {
	return "matmul"
}

func (matmulLowering MatmulLowering) ProgramKey(
	context LoweringContext,
	floatParams []float64,
	intParams []int64,
) (ProgramKey, error) {
	_ = floatParams
	_ = intParams

	if len(context.InputDTypes) != 2 || len(context.InputShapes) != 2 {
		return ProgramKey{}, &loweringError{message: "matmul requires two input tensors"}
	}

	leftShape := context.InputShapes[0]
	rightShape := context.InputShapes[1]

	if leftShape.Rank() != 2 || rightShape.Rank() != 2 || context.OutputShape.Rank() != 2 {
		return ProgramKey{}, &loweringError{message: "matmul requires rank-2 tensors"}
	}

	leftDimensions := leftShape.Dims()
	rightDimensions := rightShape.Dims()
	outputDimensions := context.OutputShape.Dims()

	if leftDimensions[1] != rightDimensions[0] {
		return ProgramKey{}, &loweringError{message: "matmul inner dimension mismatch"}
	}

	if outputDimensions[0] != leftDimensions[0] || outputDimensions[1] != rightDimensions[1] {
		return ProgramKey{}, &loweringError{message: "matmul output shape mismatch"}
	}

	return ProgramKey{
		Operation: "matmul",
		DTypes: []dtype.DType{
			context.InputDTypes[0],
			context.InputDTypes[1],
			context.OutputDType,
		},
		Shapes: []tensor.Shape{
			leftShape,
			rightShape,
			context.OutputShape,
		},
	}, nil
}

func RegisterMatmulLowerings(registry *LoweringRegistry) {
	registry.Register(MatmulLowering{})
}
