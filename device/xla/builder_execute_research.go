//go:build xla

package xla

/*
#cgo CXXFLAGS: -I${SRCDIR}/internal/bridge -std=c++17
#include "internal/bridge/core.h"
*/
import "C"

import (
	"fmt"

	"github.com/theapemachine/puter/device/xla/internal/hlo"
)

/*
ExecuteConvertUnary compiles (or loads from cache) and executes dtype-converting unary research ops.
*/
func (builder *Builder) ExecuteConvertUnary(
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

	executable, err := builder.loadConvertExecutable(bridge, programKey, context, floatParams, intParams)

	if err != nil {
		return err
	}

	return builder.recordExecute(bridge.executeUnary(C.XLAExecutableRef(executable.handle), input.bufferRef(), output.bufferRef()))
}

func (builder *Builder) loadConvertExecutable(
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
		return nil, &loweringError{message: "convert unary requires one input shape"}
	}

	count := context.InputShapes[0].Dims()[0]
	moduleName := fmt.Sprintf("puter_%s", programKey.Operation)
	scale := float32(1)
	zeroPoint := int8(0)

	if len(floatParams) > 0 {
		scale = float32(floatParams[0])
	}

	if len(intParams) > 0 {
		zeroPoint = int8(intParams[0])
	}

	var hloText string
	var err error

	switch programKey.Operation {
	case "quant_int8":
		hloText, err = hlo.RenderQuantInt8(
			moduleName,
			context.InputDTypes[0],
			count,
			scale,
			zeroPoint,
		)
	case "dequant_int8":
		hloText, err = hlo.RenderDequantInt8(
			moduleName,
			context.OutputDType,
			count,
			scale,
			zeroPoint,
		)
	case "dequant_int4":
		hloText, err = hlo.RenderDequantInt4(
			moduleName,
			context.OutputDType,
			count,
			scale,
			zeroPoint,
		)
	default:
		return nil, &loweringError{message: "unknown XLA convert operation: " + programKey.Operation}
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
ExecuteResearchUnaryParam compiles (or loads from cache) and executes parameterized unary research ops.
*/
func (builder *Builder) ExecuteResearchUnaryParam(
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

	executable, err := builder.loadResearchUnaryParamExecutable(
		bridge,
		programKey,
		context,
		floatParams,
		intParams,
	)

	if err != nil {
		return err
	}

	return builder.recordExecute(bridge.executeUnary(C.XLAExecutableRef(executable.handle), input.bufferRef(), output.bufferRef()))
}

func (builder *Builder) loadResearchUnaryParamExecutable(
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

	hloText, ok, err := renderCausalPhysicsUnaryHLO(
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
		hloText, ok, err = renderResearchUnaryParamHLO(
			moduleName,
			programKey.Operation,
			context,
			floatParams,
			intParams,
		)
	}

	if err != nil {
		return nil, err
	}

	if !ok {
		return nil, &loweringError{message: "unknown XLA research unary operation: " + programKey.Operation}
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

func renderResearchUnaryParamHLO(
	moduleName string,
	operationName string,
	context LoweringContext,
	floatParams []float64,
	intParams []int64,
) (string, bool, error) {
	if len(context.InputShapes) != 1 {
		return "", false, &loweringError{message: "research unary param requires one input shape"}
	}

	inputDimensions := context.InputShapes[0].Dims()

	switch operationName {
	case "cyclic_permute":
		count := inputDimensions[0]
		shift := 0

		if len(intParams) > 0 {
			shift = int(intParams[0])
		}

		hloText, err := hlo.RenderCyclicPermute(
			moduleName,
			context.InputDTypes[0],
			count,
			shift,
		)

		return hloText, true, err
	case "hawkes_kernel_matrix":
		eventCount := inputDimensions[0]
		alpha := float32(0)
		beta := float32(0)

		if len(floatParams) > 0 {
			alpha = float32(floatParams[0])
		}

		if len(floatParams) > 1 {
			beta = float32(floatParams[1])
		}

		hloText, err := hlo.RenderHawkesKernelMatrix(
			moduleName,
			context.InputDTypes[0],
			eventCount,
			alpha,
			beta,
		)

		return hloText, true, err
	case "hawkes_log_likelihood":
		eventCount := inputDimensions[0]
		totalT := float32(0)
		mu := float32(0)
		alpha := float32(0)
		beta := float32(0)

		if len(floatParams) > 0 {
			totalT = float32(floatParams[0])
		}

		if len(floatParams) > 1 {
			mu = float32(floatParams[1])
		}

		if len(floatParams) > 2 {
			alpha = float32(floatParams[2])
		}

		if len(floatParams) > 3 {
			beta = float32(floatParams[3])
		}

		hloText, err := hlo.RenderHawkesLogLikelihood(
			moduleName,
			context.InputDTypes[0],
			eventCount,
			totalT,
			mu,
			alpha,
			beta,
		)

		return hloText, true, err
	default:
		return "", false, nil
	}
}

func renderResearchVariadicHLO(
	moduleName string,
	operationName string,
	context LoweringContext,
	floatParams []float64,
	intParams []int64,
) (string, bool, error) {
	switch operationName {
	case "belief_update":
		if len(context.InputShapes) != 2 {
			return "", false, &loweringError{message: "belief update requires two input shapes"}
		}

		count := context.InputShapes[0].Dims()[0]
		hloText, err := hlo.RenderBeliefUpdate(moduleName, context.OutputDType, count)

		return hloText, true, err
	case "precision_weight":
		if len(context.InputShapes) != 2 {
			return "", false, &loweringError{message: "precision weight requires two input shapes"}
		}

		count := context.InputShapes[0].Dims()[0]
		hloText, err := hlo.RenderPrecisionWeight(moduleName, context.OutputDType, count)

		return hloText, true, err
	case "free_energy":
		if len(context.InputShapes) != 3 {
			return "", false, &loweringError{message: "free energy requires three input shapes"}
		}

		count := context.InputShapes[0].Dims()[0]
		hloText, err := hlo.RenderFreeEnergy(moduleName, context.OutputDType, count)

		return hloText, true, err
	case "expected_free_energy":
		if len(context.InputShapes) != 3 {
			return "", false, &loweringError{message: "expected free energy requires three input shapes"}
		}

		obsCount := context.InputShapes[0].Dims()[0]
		stateCount := context.InputShapes[2].Dims()[0]
		hloText, err := hlo.RenderExpectedFreeEnergy(
			moduleName,
			context.OutputDType,
			obsCount,
			stateCount,
		)

		return hloText, true, err
	case "update_representation":
		if len(context.InputShapes) != 3 {
			return "", false, &loweringError{message: "update representation requires three input shapes"}
		}

		outDim := int(context.InputShapes[0].Dims()[0])
		inDim := int(context.InputShapes[1].Dims()[0])
		learningRate := float32(0)

		if len(floatParams) > 0 {
			learningRate = float32(floatParams[0])
		}

		hloText, err := hlo.RenderUpdateRepresentation(
			moduleName,
			context.OutputDType,
			outDim,
			inDim,
			learningRate,
		)

		return hloText, true, err
	case "update_weights":
		if len(context.InputShapes) != 3 {
			return "", false, &loweringError{message: "update weights requires three input shapes"}
		}

		outDim := int(context.InputShapes[0].Dims()[0])
		inDim := int(context.InputShapes[1].Dims()[0])
		learningRate := float32(0)

		if len(floatParams) > 0 {
			learningRate = float32(floatParams[0])
		}

		hloText, err := hlo.RenderUpdateWeights(
			moduleName,
			context.OutputDType,
			outDim,
			inDim,
			learningRate,
		)

		return hloText, true, err
	case "hawkes_intensity":
		if len(context.InputShapes) != 2 {
			return "", false, &loweringError{message: "hawkes intensity requires two input shapes"}
		}

		eventCount := context.InputShapes[0].Dims()[0]
		queryCount := context.InputShapes[1].Dims()[0]
		mu := float32(0)
		alpha := float32(0)
		beta := float32(0)

		if len(floatParams) > 0 {
			mu = float32(floatParams[0])
		}

		if len(floatParams) > 1 {
			alpha = float32(floatParams[1])
		}

		if len(floatParams) > 2 {
			beta = float32(floatParams[2])
		}

		hloText, err := hlo.RenderHawkesIntensity(
			moduleName,
			context.OutputDType,
			eventCount,
			queryCount,
			mu,
			alpha,
			beta,
		)

		return hloText, true, err
	case "markov_blanket_partition":
		if len(context.InputShapes) != 2 {
			return "", false, &loweringError{message: "markov blanket partition requires two input shapes"}
		}

		nodeCount := context.InputShapes[0].Dims()[0]
		internalCount := context.InputShapes[1].Dims()[0]
		hloText, err := hlo.RenderMarkovBlanketPartition(moduleName, nodeCount, internalCount)

		return hloText, true, err
	default:
		return "", false, nil
	}
}
