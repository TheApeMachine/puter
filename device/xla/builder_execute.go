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
