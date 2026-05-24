//go:build xla

package xla

/*
#cgo CXXFLAGS: -I${SRCDIR}/internal/bridge -std=c++17
#include "internal/bridge/core.h"
*/
import "C"

import (
	"fmt"
	"sync"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device/xla/internal/hlo"
)

/*
ExecuteUnary compiles (or loads from cache) and executes an XLA program.
*/
func (builder *Builder) ExecuteUnary(
	bridge *xlaBridge,
	operationName string,
	context LoweringContext,
	floatParams []float64,
	intParams []int64,
	input *DeviceTensor,
	output *DeviceTensor,
) error {
	programKey, err := builder.ProgramKeyFor(operationName, context, floatParams, intParams)

	if err != nil {
		return err
	}

	executable, err := builder.loadExecutable(bridge, programKey, context, floatParams, intParams, true)

	if err != nil {
		return err
	}

	return builder.recordExecute(bridge.executeUnary(C.XLAExecutableRef(executable.handle), input.bufferRef(), output.bufferRef()))
}

/*
ExecuteBinary compiles (or loads from cache) and executes a binary XLA program.
*/
func (builder *Builder) ExecuteBinary(
	bridge *xlaBridge,
	operationName string,
	context LoweringContext,
	floatParams []float64,
	intParams []int64,
	left *DeviceTensor,
	right *DeviceTensor,
	output *DeviceTensor,
) error {
	_ = intParams
	programKey, err := builder.ProgramKeyFor(operationName, context, floatParams, intParams)

	if err != nil {
		return err
	}

	executable, err := builder.loadExecutable(bridge, programKey, context, floatParams, intParams, false)

	if err != nil {
		return err
	}

	return builder.recordExecute(bridge.executeBinary(C.XLAExecutableRef(executable.handle), left.bufferRef(), right.bufferRef(), output.bufferRef()))
}

/*
ExecuteReduction compiles (or loads from cache) and executes a vector-to-scalar reduction.
*/
func (builder *Builder) ExecuteReduction(
	bridge *xlaBridge,
	operationName string,
	context LoweringContext,
	input *DeviceTensor,
	output *DeviceTensor,
) error {
	programKey, err := builder.ProgramKeyFor(operationName, context, nil, nil)

	if err != nil {
		return err
	}

	executable, err := builder.loadReductionExecutable(bridge, programKey, context)

	if err != nil {
		return err
	}

	return builder.recordExecute(bridge.executeUnary(C.XLAExecutableRef(executable.handle), input.bufferRef(), output.bufferRef()))
}

func (builder *Builder) loadReductionExecutable(
	bridge *xlaBridge,
	programKey ProgramKey,
	context LoweringContext,
) (*CompiledExecutable, error) {
	if cached, ok := builder.CachedExecutable(programKey); ok {
		return cached, nil
	}

	reductionKind, ok := hlo.ReductionKindFromOperation(programKey.Operation)

	if !ok {
		return nil, &loweringError{message: "invalid XLA reduction operation: " + programKey.Operation}
	}

	if len(context.InputShapes) != 1 {
		return nil, &loweringError{message: "reduction requires one input shape"}
	}

	hloText, err := hlo.RenderReduction(
		fmt.Sprintf("puter_%s", programKey.Operation),
		context.OutputDType,
		context.InputShapes[0],
		reductionKind,
	)

	if err != nil {
		return nil, err
	}

	executableRef, err := bridge.compileHLO(hloText)

	if err != nil {
		return nil, err
	}

	compiled := &CompiledExecutable{
		key:    programKey,
		handle: uintptr(executableRef),
	}
	builder.RecordExecutable(programKey, compiled)
	return compiled, nil
}

/*
ExecuteDot compiles (or loads from cache) and executes a vector dot product to scalar.
*/
func (builder *Builder) ExecuteDot(
	bridge *xlaBridge,
	context LoweringContext,
	left *DeviceTensor,
	right *DeviceTensor,
	output *DeviceTensor,
) error {
	programKey, err := builder.ProgramKeyFor("dot", context, nil, nil)

	if err != nil {
		return err
	}

	executable, err := builder.loadDotExecutable(bridge, programKey, context)

	if err != nil {
		return err
	}

	return builder.recordExecute(bridge.executeBinary(C.XLAExecutableRef(executable.handle), left.bufferRef(), right.bufferRef(), output.bufferRef()))
}

func (builder *Builder) loadDotExecutable(
	bridge *xlaBridge,
	programKey ProgramKey,
	context LoweringContext,
) (*CompiledExecutable, error) {
	if cached, ok := builder.CachedExecutable(programKey); ok {
		return cached, nil
	}

	if len(context.InputShapes) != 2 {
		return nil, &loweringError{message: "dot requires two input shapes"}
	}

	hloText, err := hlo.RenderDotProduct(
		fmt.Sprintf("puter_%s", programKey.Operation),
		context.OutputDType,
		context.InputShapes[0],
		context.InputShapes[1],
	)

	if err != nil {
		return nil, err
	}

	executableRef, err := bridge.compileHLO(hloText)

	if err != nil {
		return nil, err
	}

	compiled := &CompiledExecutable{
		key:    programKey,
		handle: uintptr(executableRef),
	}
	builder.RecordExecutable(programKey, compiled)
	return compiled, nil
}

/*
ExecuteMatmul compiles (or loads from cache) and executes a rank-2 matrix multiply.
*/
func (builder *Builder) ExecuteMatmul(
	bridge *xlaBridge,
	context LoweringContext,
	left *DeviceTensor,
	right *DeviceTensor,
	output *DeviceTensor,
) error {
	programKey, err := builder.ProgramKeyFor("matmul", context, nil, nil)

	if err != nil {
		return err
	}

	executable, err := builder.loadMatmulExecutable(bridge, programKey, context)

	if err != nil {
		return err
	}

	return builder.recordExecute(bridge.executeBinary(C.XLAExecutableRef(executable.handle), left.bufferRef(), right.bufferRef(), output.bufferRef()))
}

func (builder *Builder) loadMatmulExecutable(
	bridge *xlaBridge,
	programKey ProgramKey,
	context LoweringContext,
) (*CompiledExecutable, error) {
	if cached, ok := builder.CachedExecutable(programKey); ok {
		return cached, nil
	}

	if len(context.InputShapes) != 2 {
		return nil, &loweringError{message: "matmul requires two input shapes"}
	}

	hloText, err := hlo.RenderMatmul(
		fmt.Sprintf("puter_%s", programKey.Operation),
		context.OutputDType,
		context.InputShapes[0],
		context.InputShapes[1],
		context.OutputShape,
	)

	if err != nil {
		return nil, err
	}

	executableRef, err := bridge.compileHLO(hloText)

	if err != nil {
		return nil, err
	}

	compiled := &CompiledExecutable{
		key:    programKey,
		handle: uintptr(executableRef),
	}
	builder.RecordExecutable(programKey, compiled)
	return compiled, nil
}

/*
ExecutePool compiles (or loads from cache) and executes a rank-4 pool operation.
*/
func (builder *Builder) ExecutePool(
	bridge *xlaBridge,
	operationName string,
	context LoweringContext,
	intParams []int64,
	input *DeviceTensor,
	output *DeviceTensor,
) error {
	programKey, err := builder.ProgramKeyFor(operationName, context, nil, intParams)

	if err != nil {
		return err
	}

	executable, err := builder.loadPoolExecutable(bridge, programKey, context, intParams)

	if err != nil {
		return err
	}

	return builder.recordExecute(bridge.executeUnary(C.XLAExecutableRef(executable.handle), input.bufferRef(), output.bufferRef()))
}

func (builder *Builder) loadPoolExecutable(
	bridge *xlaBridge,
	programKey ProgramKey,
	context LoweringContext,
	intParams []int64,
) (*CompiledExecutable, error) {
	if cached, ok := builder.CachedExecutable(programKey); ok {
		return cached, nil
	}

	if len(context.InputShapes) != 1 {
		return nil, &loweringError{message: "pool requires one input shape"}
	}

	inputShape := context.InputShapes[0]
	outputShape := context.OutputShape
	moduleName := fmt.Sprintf("puter_%s", programKey.Operation)

	var hloText string
	var err error

	switch programKey.Operation {
	case "max_pool2d":
		poolParams := poolParamsFromIntParams(intParams, inputShape, outputShape)
		hloText, err = hlo.RenderMaxPool2D(moduleName, context.OutputDType, inputShape, outputShape, poolParams)
	case "avg_pool2d":
		poolParams := poolParamsFromIntParams(intParams, inputShape, outputShape)
		hloText, err = hlo.RenderAvgPool2D(moduleName, context.OutputDType, inputShape, outputShape, poolParams)
	case "adaptive_max_pool2d":
		hloText, err = hlo.RenderAdaptiveMaxPool2D(moduleName, context.OutputDType, inputShape, outputShape)
	case "adaptive_avg_pool2d":
		hloText, err = hlo.RenderAdaptiveAvgPool2D(moduleName, context.OutputDType, inputShape, outputShape)
	default:
		return nil, &loweringError{message: "unknown XLA pool operation: " + programKey.Operation}
	}

	if err != nil {
		return nil, err
	}

	executableRef, err := bridge.compileHLO(hloText)

	if err != nil {
		return nil, err
	}

	compiled := &CompiledExecutable{
		key:    programKey,
		handle: uintptr(executableRef),
	}
	builder.RecordExecutable(programKey, compiled)
	return compiled, nil
}

/*
ExecuteConvolution compiles (or loads from cache) and executes convolution with bias.
*/
func (builder *Builder) ExecuteConvolution(
	bridge *xlaBridge,
	operationName string,
	context LoweringContext,
	intParams []int64,
	input *DeviceTensor,
	weight *DeviceTensor,
	bias *DeviceTensor,
	output *DeviceTensor,
) error {
	programKey, err := builder.ProgramKeyFor(operationName, context, nil, intParams)

	if err != nil {
		return err
	}

	executable, err := builder.loadConvolutionExecutable(bridge, programKey, context, intParams)

	if err != nil {
		return err
	}

	return builder.recordExecute(bridge.executeVariadic(
		C.XLAExecutableRef(executable.handle),
		[]*DeviceTensor{input, weight, bias},
		output,
	))
}

func (builder *Builder) loadConvolutionExecutable(
	bridge *xlaBridge,
	programKey ProgramKey,
	context LoweringContext,
	intParams []int64,
) (*CompiledExecutable, error) {
	if cached, ok := builder.CachedExecutable(programKey); ok {
		return cached, nil
	}

	if len(context.InputShapes) != 3 {
		return nil, &loweringError{message: "convolution requires input, weight, and bias shapes"}
	}

	moduleName := fmt.Sprintf("puter_%s", programKey.Operation)
	inputShape := context.InputShapes[0]
	weightShape := context.InputShapes[1]
	biasShape := context.InputShapes[2]
	outputShape := context.OutputShape

	var hloText string
	var err error

	switch programKey.Operation {
	case "conv2d":
		convParams, paramsErr := hlo.ConvParamsFromIntParams(intParams, 2, false)
		if paramsErr != nil {
			return nil, paramsErr
		}

		hloText, err = hlo.RenderConv2D(
			moduleName,
			context.OutputDType,
			inputShape,
			weightShape,
			biasShape,
			outputShape,
			convParams,
		)
	case "conv1d":
		convParams, paramsErr := hlo.ConvParamsFromIntParams(intParams, 1, false)
		if paramsErr != nil {
			return nil, paramsErr
		}

		hloText, err = hlo.RenderConv1D(
			moduleName,
			context.OutputDType,
			inputShape,
			weightShape,
			biasShape,
			outputShape,
			convParams,
		)
	case "conv3d":
		convParams, paramsErr := hlo.ConvParamsFromIntParams(intParams, 3, false)
		if paramsErr != nil {
			return nil, paramsErr
		}

		hloText, err = hlo.RenderConv3D(
			moduleName,
			context.OutputDType,
			inputShape,
			weightShape,
			biasShape,
			outputShape,
			convParams,
		)
	case "conv_transpose2d":
		convParams, paramsErr := hlo.ConvParamsFromIntParams(intParams, 2, true)
		if paramsErr != nil {
			return nil, paramsErr
		}

		hloText, err = hlo.RenderConvTranspose2D(
			moduleName,
			context.OutputDType,
			inputShape,
			weightShape,
			biasShape,
			outputShape,
			convParams,
		)
	default:
		return nil, &loweringError{message: "unknown XLA convolution operation: " + programKey.Operation}
	}

	if err != nil {
		return nil, err
	}

	executableRef, err := bridge.compileHLO(hloText)

	if err != nil {
		return nil, err
	}

	compiled := &CompiledExecutable{
		key:    programKey,
		handle: uintptr(executableRef),
	}
	builder.RecordExecutable(programKey, compiled)
	return compiled, nil
}

/*
ExecuteVariadic compiles (or loads from cache) and executes a multi-input XLA program.
*/
func (builder *Builder) ExecuteVariadic(
	bridge *xlaBridge,
	operationName string,
	context LoweringContext,
	floatParams []float64,
	intParams []int64,
	inputs []*DeviceTensor,
	output *DeviceTensor,
) error {
	programKey, err := builder.ProgramKeyFor(operationName, context, floatParams, intParams)

	if err != nil {
		return err
	}

	executable, err := builder.loadVariadicExecutable(bridge, programKey, context, floatParams, intParams)

	if err != nil {
		return err
	}

	return builder.recordExecute(bridge.executeVariadic(C.XLAExecutableRef(executable.handle), inputs, output))
}

/*
ExecutePairLoss compiles (or loads from cache) and executes a pair-wise loss to scalar.
*/
func (builder *Builder) ExecutePairLoss(
	bridge *xlaBridge,
	operationName string,
	context LoweringContext,
	predictions *DeviceTensor,
	targets *DeviceTensor,
	output *DeviceTensor,
) error {
	programKey, err := builder.ProgramKeyFor(operationName, context, nil, nil)

	if err != nil {
		return err
	}

	executable, err := builder.loadPairLossExecutable(bridge, programKey, context)

	if err != nil {
		return err
	}

	return builder.recordExecute(bridge.executeBinary(C.XLAExecutableRef(executable.handle), predictions.bufferRef(), targets.bufferRef(), output.bufferRef()))
}

func (builder *Builder) loadPairLossExecutable(
	bridge *xlaBridge,
	programKey ProgramKey,
	context LoweringContext,
) (*CompiledExecutable, error) {
	if cached, ok := builder.CachedExecutable(programKey); ok {
		return cached, nil
	}

	lossKind, ok := hlo.PairLossKindFromOperation(programKey.Operation)

	if !ok {
		return nil, &loweringError{message: "invalid XLA pair loss operation: " + programKey.Operation}
	}

	if len(context.InputShapes) != 2 {
		return nil, &loweringError{message: "pair loss requires two input shapes"}
	}

	hloText, err := hlo.RenderPairLoss(
		fmt.Sprintf("puter_%s", programKey.Operation),
		context.OutputDType,
		context.InputShapes[0],
		lossKind,
	)

	if err != nil {
		return nil, err
	}

	executableRef, err := bridge.compileHLO(hloText)

	if err != nil {
		return nil, err
	}

	compiled := &CompiledExecutable{
		key:    programKey,
		handle: uintptr(executableRef),
	}
	builder.RecordExecutable(programKey, compiled)
	return compiled, nil
}

/*
ExecuteCrossEntropy compiles (or loads from cache) and executes cross entropy to scalar.
*/
func (builder *Builder) ExecuteCrossEntropy(
	bridge *xlaBridge,
	context LoweringContext,
	logits *DeviceTensor,
	targets *DeviceTensor,
	output *DeviceTensor,
) error {
	programKey, err := builder.ProgramKeyFor("cross_entropy", context, nil, nil)

	if err != nil {
		return err
	}

	executable, err := builder.loadCrossEntropyExecutable(bridge, programKey, context)

	if err != nil {
		return err
	}

	return builder.recordExecute(bridge.executeVariadic(
		C.XLAExecutableRef(executable.handle),
		[]*DeviceTensor{logits, targets},
		output,
	))
}

func (builder *Builder) loadCrossEntropyExecutable(
	bridge *xlaBridge,
	programKey ProgramKey,
	context LoweringContext,
) (*CompiledExecutable, error) {
	if cached, ok := builder.CachedExecutable(programKey); ok {
		return cached, nil
	}

	if len(context.InputShapes) != 2 {
		return nil, &loweringError{message: "cross entropy requires logits and target shapes"}
	}

	logitsShape := context.InputShapes[0]
	logitsDimensions := logitsShape.Dims()

	if len(logitsDimensions) != 2 {
		return nil, &loweringError{message: "cross entropy logits must be rank-2"}
	}

	hloText, err := hlo.RenderCrossEntropy(
		fmt.Sprintf("puter_%s", programKey.Operation),
		context.OutputDType,
		logitsDimensions[0],
		logitsDimensions[1],
	)

	if err != nil {
		return nil, err
	}

	executableRef, err := bridge.compileHLO(hloText)

	if err != nil {
		return nil, err
	}

	compiled := &CompiledExecutable{
		key:    programKey,
		handle: uintptr(executableRef),
	}
	builder.RecordExecutable(programKey, compiled)
	return compiled, nil
}

/*
ExecuteRoPE compiles (or loads from cache) and executes rotary positional embedding.
*/
func (builder *Builder) ExecuteRoPE(
	bridge *xlaBridge,
	context LoweringContext,
	floatParams []float64,
	intParams []int64,
	input *DeviceTensor,
	output *DeviceTensor,
) error {
	programKey, err := builder.ProgramKeyFor("rope", context, floatParams, intParams)

	if err != nil {
		return err
	}

	executable, err := builder.loadRoPEExecutable(bridge, programKey, context, floatParams, intParams)

	if err != nil {
		return err
	}

	return builder.recordExecute(bridge.executeUnary(C.XLAExecutableRef(executable.handle), input.bufferRef(), output.bufferRef()))
}

func (builder *Builder) loadRoPEExecutable(
	bridge *xlaBridge,
	programKey ProgramKey,
	context LoweringContext,
	floatParams []float64,
	intParams []int64,
) (*CompiledExecutable, error) {
	if cached, ok := builder.CachedExecutable(programKey); ok {
		return cached, nil
	}

	if len(context.InputShapes) != 1 {
		return nil, &loweringError{message: "rope requires one input shape"}
	}

	inputDimensions := context.InputShapes[0].Dims()

	if len(inputDimensions) != 3 {
		return nil, &loweringError{message: "rope requires rank-3 input"}
	}

	baseFreq := 10000.0
	startPosition := 0

	if len(floatParams) > 0 {
		baseFreq = floatParams[0]
	}

	if len(intParams) > 0 {
		startPosition = int(intParams[0])
	}

	hloText, err := hlo.RenderRoPE(
		fmt.Sprintf("puter_%s", programKey.Operation),
		context.OutputDType,
		inputDimensions[0],
		inputDimensions[1],
		inputDimensions[2],
		baseFreq,
		startPosition,
	)

	if err != nil {
		return nil, err
	}

	executableRef, err := bridge.compileHLO(hloText)

	if err != nil {
		return nil, err
	}

	compiled := &CompiledExecutable{
		key:    programKey,
		handle: uintptr(executableRef),
	}
	builder.RecordExecutable(programKey, compiled)
	return compiled, nil
}

func renderScaledDotProductAttention(
	moduleName string,
	context LoweringContext,
	intParams []int64,
) (string, error) {
	if len(context.InputShapes) != 3 {
		return "", &loweringError{message: "scaled dot product attention requires three input shapes"}
	}

	queryShape := context.InputShapes[0]
	keyShape := context.InputShapes[1]
	valueShape := context.InputShapes[2]
	seqQ := queryShape.Dims()[0]
	seqK := keyShape.Dims()[0]
	depth := queryShape.Dims()[1]
	valueDim := valueShape.Dims()[1]
	causal := false

	if len(intParams) > 0 && intParams[0] != 0 {
		causal = true
	}

	return hlo.RenderScaledDotProductAttention(
		moduleName,
		context.OutputDType,
		seqQ,
		seqK,
		depth,
		valueDim,
		causal,
	)
}

func renderMultiHeadAttention(
	moduleName string,
	context LoweringContext,
	floatParams []float64,
	intParams []int64,
) (string, error) {
	if len(context.InputShapes) != 3 {
		return "", &loweringError{message: "multi head attention requires three input shapes"}
	}

	queryShape := context.InputShapes[0]
	keyShape := context.InputShapes[1]
	seqQ := queryShape.Dims()[0]
	seqK := keyShape.Dims()[0]
	headDim := 0
	numHeads := 0
	kvHeads := 0
	causal := false
	windowSize := 0
	alibiSlope := float32(0)

	if len(intParams) > 0 {
		numHeads = int(intParams[0])
	}

	if len(intParams) > 1 {
		headDim = int(intParams[1])
	}

	if len(intParams) > 2 {
		kvHeads = int(intParams[2])
	}

	if len(intParams) > 3 && intParams[3] != 0 {
		causal = true
	}

	if len(intParams) > 4 {
		windowSize = int(intParams[4])
	}

	if len(floatParams) > 0 {
		alibiSlope = float32(floatParams[0])
	}

	if numHeads == 0 || headDim == 0 {
		return "", &loweringError{message: "multi head attention requires head metadata"}
	}

	return hlo.RenderMultiHeadAttention(
		moduleName,
		context.OutputDType,
		seqQ,
		seqK,
		numHeads,
		kvHeads,
		headDim,
		causal,
		windowSize,
		alibiSlope,
	)
}

func (builder *Builder) loadVariadicExecutable(
	bridge *xlaBridge,
	programKey ProgramKey,
	context LoweringContext,
	floatParams []float64,
	intParams []int64,
) (*CompiledExecutable, error) {
	if cached, ok := builder.CachedExecutable(programKey); ok {
		return cached, nil
	}

	moduleName := fmt.Sprintf("puter_%s", programKey.Operation)
	inputShape := context.InputShapes[0]

	var hloText string
	var err error

	switch programKey.Operation {
	case "layer_norm":
		hloText, err = hlo.RenderLayerNorm(moduleName, context.OutputDType, inputShape)
	case "rms_norm":
		hloText, err = hlo.RenderRMSNorm(moduleName, context.OutputDType, inputShape)
	case "batch_norm_eval":
		hloText, err = hlo.RenderBatchNormEval(moduleName, context.OutputDType, inputShape)
	case "instance_norm":
		hloText, err = hlo.RenderInstanceNorm(moduleName, context.OutputDType, inputShape)
	case "group_norm":
		groups := 1

		if len(intParams) > 0 {
			groups = int(intParams[0])
		}

		hloText, err = hlo.RenderGroupNorm(moduleName, context.OutputDType, inputShape, groups)
	case "rope_pairs":
		headDim := inputShape.Dims()[0]
		hloText, err = hlo.RenderRoPEPairs(moduleName, context.OutputDType, headDim)
	case "scaled_dot_product_attention":
		hloText, err = renderScaledDotProductAttention(moduleName, context, intParams)
	case "multi_head_attention":
		hloText, err = renderMultiHeadAttention(moduleName, context, floatParams, intParams)
	case "alibi_bias":
		hloText, err = renderALiBiBias(moduleName, context)
	case "embedding_lookup":
		hloText, err = renderEmbeddingLookup(moduleName, context)
	case "embedding_bag":
		hloText, err = renderEmbeddingBag(moduleName, context)
	case "matmul_bias_gelu":
		hloText, err = hlo.RenderMatmulBiasGelu(
			moduleName,
			context.OutputDType,
			context.InputShapes[0],
			context.InputShapes[1],
			context.OutputShape,
		)
	case "layernorm_residual":
		hloText, err = hlo.RenderLayernormResidual(moduleName, context.OutputDType, inputShape)
	default:
		var ok bool
		hloText, ok, err = renderCausalPhysicsVariadicHLO(
			moduleName,
			programKey.Operation,
			context,
			floatParams,
			intParams,
		)

		if err != nil {
			return nil, err
		}

		if !ok {
			hloText, ok, err = renderResearchVariadicHLO(
				moduleName,
				programKey.Operation,
				context,
				floatParams,
				intParams,
			)
		}

		if !ok {
			return nil, &loweringError{message: "unknown XLA variadic operation: " + programKey.Operation}
		}
	}

	_ = floatParams

	if err != nil {
		return nil, err
	}

	executableRef, err := bridge.compileHLO(hloText)

	if err != nil {
		return nil, err
	}

	compiled := &CompiledExecutable{
		key:    programKey,
		handle: uintptr(executableRef),
	}
	builder.RecordExecutable(programKey, compiled)
	return compiled, nil
}

/*
ExecuteNullary compiles (or loads from cache) and executes an input-free XLA program.
*/
func (builder *Builder) ExecuteNullary(
	bridge *xlaBridge,
	operationName string,
	context LoweringContext,
	intParams []int64,
	output *DeviceTensor,
) error {
	programKey, err := builder.ProgramKeyFor(operationName, context, nil, intParams)

	if err != nil {
		return err
	}

	executable, err := builder.loadNullaryExecutable(bridge, programKey, context, intParams)

	if err != nil {
		return err
	}

	return builder.recordExecute(bridge.executeNullary(C.XLAExecutableRef(executable.handle), output))
}

func (builder *Builder) loadNullaryExecutable(
	bridge *xlaBridge,
	programKey ProgramKey,
	context LoweringContext,
	intParams []int64,
) (*CompiledExecutable, error) {
	if cached, ok := builder.CachedExecutable(programKey); ok {
		return cached, nil
	}

	moduleName := fmt.Sprintf("puter_%s", programKey.Operation)
	var hloText string
	var err error

	switch programKey.Operation {
	case "causal_mask":
		seqQ, seqK, shapeErr := matrixDimensions(intParams, context.OutputShape)

		if shapeErr != nil {
			return nil, shapeErr
		}

		hloText, err = hlo.RenderCausalMask(moduleName, context.OutputDType, seqQ, seqK)
	default:
		return nil, &loweringError{message: "unknown XLA nullary operation: " + programKey.Operation}
	}

	if err != nil {
		return nil, err
	}

	executableRef, err := bridge.compileHLO(hloText)

	if err != nil {
		return nil, err
	}

	compiled := &CompiledExecutable{
		key:    programKey,
		handle: uintptr(executableRef),
	}
	builder.RecordExecutable(programKey, compiled)
	return compiled, nil
}

/*
ExecuteDropout compiles (or loads from cache) and executes inverted dropout.
*/
func (builder *Builder) ExecuteDropout(
	bridge *xlaBridge,
	context LoweringContext,
	rate float32,
	seed uint64,
	input *DeviceTensor,
	output *DeviceTensor,
) error {
	floatParams := []float64{float64(rate)}
	intParams := []int64{
		int64(uint32(seed)),
		int64(uint32(seed >> 32)),
	}

	programKey, err := builder.ProgramKeyFor("dropout", context, floatParams, intParams)

	if err != nil {
		return err
	}

	executable, err := builder.loadDropoutExecutable(bridge, programKey, context, rate, seed)

	if err != nil {
		return err
	}

	return builder.recordExecute(bridge.executeUnary(C.XLAExecutableRef(executable.handle), input.bufferRef(), output.bufferRef()))
}

func (builder *Builder) loadDropoutExecutable(
	bridge *xlaBridge,
	programKey ProgramKey,
	context LoweringContext,
	rate float32,
	seed uint64,
) (*CompiledExecutable, error) {
	if cached, ok := builder.CachedExecutable(programKey); ok {
		return cached, nil
	}

	if len(context.InputShapes) != 1 {
		return nil, &loweringError{message: "dropout requires one input shape"}
	}

	hloText, err := hlo.RenderDropout(
		fmt.Sprintf("puter_%s", programKey.Operation),
		context.OutputDType,
		context.InputShapes[0],
		rate,
		seed,
	)

	if err != nil {
		return nil, err
	}

	executableRef, err := bridge.compileHLO(hloText)

	if err != nil {
		return nil, err
	}

	compiled := &CompiledExecutable{
		key:    programKey,
		handle: uintptr(executableRef),
	}
	builder.RecordExecutable(programKey, compiled)
	return compiled, nil
}

/*
ExecuteGreedySample compiles (or loads from cache) and executes argmax to scalar int32.
*/
func (builder *Builder) ExecuteGreedySample(
	bridge *xlaBridge,
	context LoweringContext,
	input *DeviceTensor,
	output *DeviceTensor,
) error {
	programKey, err := builder.ProgramKeyFor("greedy_sample", context, nil, nil)

	if err != nil {
		return err
	}

	executable, err := builder.loadGreedySampleExecutable(bridge, programKey, context)

	if err != nil {
		return err
	}

	return builder.recordExecute(bridge.executeUnary(C.XLAExecutableRef(executable.handle), input.bufferRef(), output.bufferRef()))
}

/*
ExecuteSoftmaxSort compiles (or loads from cache) and executes temperature softmax with descending sort.
The output stacks sorted probabilities and sorted indices as float32 lanes.
*/
func (builder *Builder) ExecuteSoftmaxSort(
	bridge *xlaBridge,
	context LoweringContext,
	floatParams []float64,
	input *DeviceTensor,
	output *DeviceTensor,
) error {
	programKey, err := builder.ProgramKeyFor("softmax_sort", context, floatParams, nil)

	if err != nil {
		return err
	}

	executable, err := builder.loadSoftmaxSortExecutable(bridge, programKey, context, floatParams)

	if err != nil {
		return err
	}

	return builder.recordExecute(bridge.executeUnary(C.XLAExecutableRef(executable.handle), input.bufferRef(), output.bufferRef()))
}

func (builder *Builder) loadSoftmaxSortExecutable(
	bridge *xlaBridge,
	programKey ProgramKey,
	context LoweringContext,
	floatParams []float64,
) (*CompiledExecutable, error) {
	if cached, ok := builder.CachedExecutable(programKey); ok {
		return cached, nil
	}

	if len(context.InputShapes) != 1 {
		return nil, &loweringError{message: "softmax sort requires one input shape"}
	}

	vocabSize := context.InputShapes[0].Dims()[0]
	temperature := float32(1)

	if len(floatParams) > 0 {
		temperature = float32(floatParams[0])
	}

	hloText, err := hlo.RenderSoftmaxSort(
		fmt.Sprintf("puter_%s", programKey.Operation),
		context.InputDTypes[0],
		vocabSize,
		temperature,
	)

	if err != nil {
		return nil, err
	}

	executableRef, err := bridge.compileHLO(hloText)

	if err != nil {
		return nil, err
	}

	compiled := &CompiledExecutable{
		key:    programKey,
		handle: uintptr(executableRef),
	}
	builder.RecordExecutable(programKey, compiled)
	return compiled, nil
}

func (builder *Builder) loadGreedySampleExecutable(
	bridge *xlaBridge,
	programKey ProgramKey,
	context LoweringContext,
) (*CompiledExecutable, error) {
	if cached, ok := builder.CachedExecutable(programKey); ok {
		return cached, nil
	}

	if len(context.InputShapes) != 1 {
		return nil, &loweringError{message: "greedy sample requires one input shape"}
	}

	vocabSize := context.InputShapes[0].Dims()[0]

	hloText, err := hlo.RenderGreedySample(
		fmt.Sprintf("puter_%s", programKey.Operation),
		context.InputDTypes[0],
		vocabSize,
	)

	if err != nil {
		return nil, err
	}

	executableRef, err := bridge.compileHLO(hloText)

	if err != nil {
		return nil, err
	}

	compiled := &CompiledExecutable{
		key:    programKey,
		handle: uintptr(executableRef),
	}
	builder.RecordExecutable(programKey, compiled)
	return compiled, nil
}

func renderALiBiBias(
	moduleName string,
	context LoweringContext,
) (string, error) {
	if len(context.InputShapes) != 2 {
		return "", &loweringError{message: "ALiBi bias requires scores and slope shapes"}
	}

	scoreDimensions := context.InputShapes[0].Dims()

	if len(scoreDimensions) != 2 {
		return "", &loweringError{message: "ALiBi scores must be rank-2"}
	}

	return hlo.RenderALiBiBias(
		moduleName,
		context.OutputDType,
		scoreDimensions[0],
		scoreDimensions[1],
	)
}

func renderEmbeddingLookup(
	moduleName string,
	context LoweringContext,
) (string, error) {
	if len(context.InputShapes) != 2 {
		return "", &loweringError{message: "embedding lookup requires table and indices shapes"}
	}

	tableDimensions := context.InputShapes[0].Dims()
	indexCount := context.InputShapes[1].Dims()[0]

	if len(tableDimensions) != 2 {
		return "", &loweringError{message: "embedding table must be rank-2"}
	}

	return hlo.RenderEmbeddingLookup(
		moduleName,
		context.OutputDType,
		tableDimensions[0],
		tableDimensions[1],
		indexCount,
	)
}

func renderEmbeddingBag(
	moduleName string,
	context LoweringContext,
) (string, error) {
	if len(context.InputShapes) != 3 {
		return "", &loweringError{message: "embedding bag requires table, indices, and offsets shapes"}
	}

	tableDimensions := context.InputShapes[0].Dims()
	indexCount := context.InputShapes[1].Dims()[0]
	bagCount := context.InputShapes[2].Dims()[0]

	if len(tableDimensions) != 2 {
		return "", &loweringError{message: "embedding table must be rank-2"}
	}

	return hlo.RenderEmbeddingBag(
		moduleName,
		context.OutputDType,
		tableDimensions[0],
		tableDimensions[1],
		bagCount,
		indexCount,
	)
}

func matrixDimensions(intParams []int64, outputShape tensor.Shape) (int, int, error) {
	outputDimensions := outputShape.Dims()

	if len(outputDimensions) == 2 {
		return outputDimensions[0], outputDimensions[1], nil
	}

	if len(intParams) >= 2 {
		return int(intParams[0]), int(intParams[1]), nil
	}

	return 0, 0, &loweringError{message: "matrix operation requires rank-2 output or int params"}
}

func poolParamsFromIntParams(
	intParams []int64,
	inputShape tensor.Shape,
	outputShape tensor.Shape,
) hlo.PoolParams {
	inputDimensions := inputShape.Dims()
	outputDimensions := outputShape.Dims()

	kernelH := int64(0)
	kernelW := int64(0)
	strideH := int64(0)
	strideW := int64(0)
	paddingH := int64(0)
	paddingW := int64(0)

	if len(intParams) > 0 {
		kernelH = intParams[0]
	}

	if len(intParams) > 1 {
		kernelW = intParams[1]
	}

	if len(intParams) > 2 {
		strideH = intParams[2]
	}

	if len(intParams) > 3 {
		strideW = intParams[3]
	}

	if len(intParams) > 4 {
		paddingH = intParams[4]
	}

	if len(intParams) > 5 {
		paddingW = intParams[5]
	}

	return hlo.ResolvePoolParams(
		kernelH, kernelW, strideH, strideW, paddingH, paddingW,
		int64(inputDimensions[2]), int64(inputDimensions[3]),
		int64(outputDimensions[2]), int64(outputDimensions[3]),
	)
}

func (builder *Builder) loadExecutable(
	bridge *xlaBridge,
	programKey ProgramKey,
	context LoweringContext,
	floatParams []float64,
	intParams []int64,
	unary bool,
) (*CompiledExecutable, error) {
	if cached, ok := builder.CachedExecutable(programKey); ok {
		return cached, nil
	}

	moduleBuilder, err := hlo.NewModuleBuilder(
		fmt.Sprintf("puter_%s", programKey.Operation),
		context.OutputDType,
		context.OutputShape,
	)

	if err != nil {
		return nil, err
	}

	var hloText string

	hloText, err = moduleBuilder.RenderProgram(programKey.Operation, floatParams, intParams)

	if err != nil {
		return nil, err
	}

	_ = unary

	executableRef, err := bridge.compileHLO(hloText)

	if err != nil {
		return nil, err
	}

	compiled := &CompiledExecutable{
		key:    programKey,
		handle: uintptr(executableRef),
	}
	builder.RecordExecutable(programKey, compiled)
	return compiled, nil
}

var builderMutex sync.Mutex

/*
CompiledExecutableHandle stores the PJRT loaded executable handle.
*/
func (compiledExecutable *CompiledExecutable) Close(bridge *xlaBridge) {
	if compiledExecutable.handle != 0 && bridge != nil {
		bridge.releaseExecutable(C.XLAExecutableRef(compiledExecutable.handle))
		compiledExecutable.handle = 0
	}
}

/*
ShapeFromCount builds a dense 1-D shape for count-element launches.
*/
func ShapeFromCount(count int) (tensor.Shape, error) {
	if count <= 0 {
		return tensor.Shape{}, &loweringError{message: "invalid XLA element count"}
	}

	return tensor.NewShape([]int{count})
}

/*
ShapeFromMatmul builds dense rank-2 shapes for GEMM launches.
*/
func ShapeFromMatmul(rows, inner, cols int) (tensor.Shape, tensor.Shape, tensor.Shape, error) {
	if rows <= 0 || inner <= 0 || cols <= 0 {
		return tensor.Shape{}, tensor.Shape{}, tensor.Shape{}, &loweringError{message: "invalid XLA matmul dimensions"}
	}

	leftShape, err := tensor.NewShape([]int{rows, inner})

	if err != nil {
		return tensor.Shape{}, tensor.Shape{}, tensor.Shape{}, err
	}

	rightShape, err := tensor.NewShape([]int{inner, cols})

	if err != nil {
		return tensor.Shape{}, tensor.Shape{}, tensor.Shape{}, err
	}

	outputShape, err := tensor.NewShape([]int{rows, cols})

	if err != nil {
		return tensor.Shape{}, tensor.Shape{}, tensor.Shape{}, err
	}

	return leftShape, rightShape, outputShape, nil
}

/*
ShapeFromNCHW builds a dense rank-4 NCHW shape.
*/
func ShapeFromNCHW(batch, channels, height, width int) (tensor.Shape, error) {
	if batch <= 0 || channels <= 0 || height <= 0 || width <= 0 {
		return tensor.Shape{}, &loweringError{message: "invalid XLA NCHW shape"}
	}

	return tensor.NewShape([]int{batch, channels, height, width})
}

/*
ShapeFromBCS builds a dense rank-3 BCS shape.
*/
func ShapeFromBCS(batch, channels, spatial int) (tensor.Shape, error) {
	if batch <= 0 || channels <= 0 || spatial <= 0 {
		return tensor.Shape{}, &loweringError{message: "invalid XLA BCS shape"}
	}

	return tensor.NewShape([]int{batch, channels, spatial})
}

/*
ShapeFromRowsCols builds a dense rank-2 matrix shape.
*/
func ShapeFromRowsCols(rows, cols int) (tensor.Shape, error) {
	if rows <= 0 || cols <= 0 {
		return tensor.Shape{}, &loweringError{message: "invalid XLA row matrix shape"}
	}

	return tensor.NewShape([]int{rows, cols})
}

/*
ShapeFromVector builds a dense rank-1 shape.
*/
func ShapeFromVector(count int) (tensor.Shape, error) {
	if count <= 0 {
		return tensor.Shape{}, &loweringError{message: "invalid XLA vector shape"}
	}

	return tensor.NewShape([]int{count})
}

/*
LoweringContextForUnary builds lowering metadata for same-shape unary ops.
*/
func LoweringContextForUnary(
	elementFormat dtype.DType,
	shape tensor.Shape,
) LoweringContext {
	return LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{elementFormat},
		InputShapes: []tensor.Shape{shape},
		OutputDType: elementFormat,
		OutputShape: shape,
	}
}

/*
LoweringContextForBinary builds lowering metadata for binary ops with broadcast.
*/
func LoweringContextForBinary(
	elementFormat dtype.DType,
	leftShape tensor.Shape,
	rightShape tensor.Shape,
	outputShape tensor.Shape,
) LoweringContext {
	return LoweringContext{
		Target:      DefaultBuilderTarget,
		InputDTypes: []dtype.DType{elementFormat, elementFormat},
		InputShapes: []tensor.Shape{leftShape, rightShape},
		OutputDType: elementFormat,
		OutputShape: outputShape,
	}
}
