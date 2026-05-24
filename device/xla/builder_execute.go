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

	return bridge.executeUnary(C.XLAExecutableRef(executable.handle), input.bufferRef(), output.bufferRef())
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

	return bridge.executeBinary(C.XLAExecutableRef(executable.handle), left.bufferRef(), right.bufferRef(), output.bufferRef())
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

	return bridge.executeUnary(C.XLAExecutableRef(executable.handle), input.bufferRef(), output.bufferRef())
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

	return bridge.executeBinary(C.XLAExecutableRef(executable.handle), left.bufferRef(), right.bufferRef(), output.bufferRef())
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

	return bridge.executeBinary(C.XLAExecutableRef(executable.handle), left.bufferRef(), right.bufferRef(), output.bufferRef())
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

	return bridge.executeUnary(C.XLAExecutableRef(executable.handle), input.bufferRef(), output.bufferRef())
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

	return bridge.executeVariadic(C.XLAExecutableRef(executable.handle), inputs, output)
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
	default:
		return nil, &loweringError{message: "unknown XLA variadic operation: " + programKey.Operation}
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
