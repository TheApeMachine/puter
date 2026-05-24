//go:build xla

package xla

/*
#cgo CXXFLAGS: -I${SRCDIR}/internal/bridge -std=c++17
#include "internal/bridge/core.h"
*/
import "C"

import (
	"github.com/theapemachine/puter/device/xla/internal/hlo"
)

func renderCausalPhysicsVariadicHLO(
	moduleName string,
	operationName string,
	context LoweringContext,
	floatParams []float64,
	intParams []int64,
) (string, bool, error) {
	switch operationName {
	case "cate":
		if len(context.InputShapes) != 2 {
			return "", false, &loweringError{message: "cate requires two input shapes"}
		}

		count := context.InputShapes[0].Dims()[0]
		hloText, err := hlo.RenderCATE(moduleName, context.OutputDType, count)

		return hloText, true, err
	case "counterfactual":
		if len(context.InputShapes) != 3 {
			return "", false, &loweringError{message: "counterfactual requires three input shapes"}
		}

		count := context.InputShapes[0].Dims()[0]
		slope := float32(0)

		if len(floatParams) > 0 {
			slope = float32(floatParams[0])
		}

		hloText, err := hlo.RenderCounterfactual(moduleName, context.OutputDType, count, slope)

		return hloText, true, err
	case "backdoor_adjustment":
		if len(context.InputShapes) != 2 || len(intParams) < 3 {
			return "", false, &loweringError{message: "backdoor adjustment requires shapes and xyz counts"}
		}

		hloText, err := hlo.RenderBackdoorAdjustment(
			moduleName,
			context.OutputDType,
			int(intParams[0]),
			int(intParams[1]),
			int(intParams[2]),
		)

		return hloText, true, err
	case "do_intervene":
		if len(context.InputShapes) != 2 || len(intParams) < 2 {
			return "", false, &loweringError{message: "do intervene requires node and intervened counts"}
		}

		hloText, err := hlo.RenderDoIntervene(
			moduleName,
			context.InputDTypes[0],
			int(intParams[0]),
			int(intParams[1]),
		)

		return hloText, true, err
	case "frontdoor_adjustment":
		if len(context.InputShapes) != 3 || len(intParams) < 3 {
			return "", false, &loweringError{message: "frontdoor adjustment requires shapes and counts"}
		}

		hloText, err := hlo.RenderFrontdoorAdjustment(
			moduleName,
			context.OutputDType,
			int(intParams[0]),
			int(intParams[1]),
			int(intParams[2]),
		)

		return hloText, true, err
	case "iv_estimate":
		if len(context.InputShapes) != 3 {
			return "", false, &loweringError{message: "iv estimate requires three input shapes"}
		}

		count := context.InputShapes[0].Dims()[0]
		hloText, err := hlo.RenderIVEstimate(moduleName, context.OutputDType, count)

		return hloText, true, err
	case "markov_flow":
		if len(context.InputShapes) != 2 || len(intParams) < 2 {
			return "", false, &loweringError{message: "markov flow requires counts and target label"}
		}

		nodeCount := int(intParams[0])
		targetLabel := int32(intParams[1])
		hloText, err := hlo.RenderMarkovFlow(
			moduleName,
			context.OutputDType,
			nodeCount,
			targetLabel,
		)

		return hloText, true, err
	case "madelung_continuity":
		if len(context.InputShapes) != 2 {
			return "", false, &loweringError{message: "madelung continuity requires density and velocity"}
		}

		count := context.InputShapes[0].Dims()[0]
		invTwoDx := float32(0)

		if len(floatParams) > 0 {
			invTwoDx = float32(floatParams[0])
		}

		hloText, err := hlo.RenderMadelungContinuity(moduleName, context.OutputDType, count, invTwoDx)

		return hloText, true, err
	case "fft1d":
		if len(context.InputShapes) != 2 {
			return "", false, &loweringError{message: "fft1d requires real and imag inputs"}
		}

		count := context.InputShapes[0].Dims()[0]
		inverse := len(intParams) > 0 && intParams[0] != 0
		hloText, err := hlo.RenderFFT1D(moduleName, count, inverse)

		return hloText, true, err
	default:
		return "", false, nil
	}
}

func renderCausalPhysicsUnaryHLO(
	moduleName string,
	operationName string,
	context LoweringContext,
	floatParams []float64,
	intParams []int64,
) (string, bool, error) {
	if len(context.InputShapes) != 1 {
		return "", false, &loweringError{message: "causal physics unary requires one input shape"}
	}

	count := context.InputShapes[0].Dims()

	switch operationName {
	case "dag_markov_factorization":
		hloText, err := hlo.RenderDAGMarkovFactorization(
			moduleName,
			context.InputDTypes[0],
			count[0],
		)

		return hloText, true, err
	case "cholesky":
		if len(count) != 2 || count[0] != count[1] {
			return "", false, &loweringError{message: "cholesky requires square matrix input"}
		}

		hloText, err := hlo.RenderCholesky(moduleName, context.InputDTypes[0], count[0])

		return hloText, true, err
	case "grad1d":
		invTwoDx := float32(0)

		if len(floatParams) > 0 {
			invTwoDx = float32(floatParams[0])
		}

		hloText, err := hlo.RenderGrad1D(moduleName, context.InputDTypes[0], count[0], invTwoDx)

		return hloText, true, err
	case "laplacian1d":
		invH2 := float32(0)

		if len(floatParams) > 0 {
			invH2 = float32(floatParams[0])
		}

		hloText, err := hlo.RenderLaplacian1D(moduleName, context.InputDTypes[0], count[0], invH2)

		return hloText, true, err
	case "laplacian2d":
		if len(count) != 2 {
			return "", false, &loweringError{message: "laplacian2d requires rank-2 input"}
		}

		invH2 := float32(0)

		if len(floatParams) > 0 {
			invH2 = float32(floatParams[0])
		}

		hloText, err := hlo.RenderLaplacian2D(
			moduleName,
			context.InputDTypes[0],
			count[0],
			count[1],
			invH2,
		)

		return hloText, true, err
	case "laplacian3d":
		if len(count) != 3 {
			return "", false, &loweringError{message: "laplacian3d requires rank-3 input"}
		}

		invH2 := float32(0)

		if len(floatParams) > 0 {
			invH2 = float32(floatParams[0])
		}

		hloText, err := hlo.RenderLaplacian3D(
			moduleName,
			context.InputDTypes[0],
			count[0],
			count[1],
			count[2],
			invH2,
		)

		return hloText, true, err
	case "laplacian4":
		invDen := float32(0)

		if len(floatParams) > 0 {
			invDen = float32(floatParams[0])
		}

		hloText, err := hlo.RenderLaplacian4(moduleName, context.InputDTypes[0], count[0], invDen)

		return hloText, true, err
	case "central_difference_interior":
		scale := float32(0)

		if len(floatParams) > 0 {
			scale = float32(floatParams[0])
		}

		hloText, err := hlo.RenderCentralDifferenceInterior(
			moduleName,
			context.InputDTypes[0],
			count[0],
			scale,
		)

		return hloText, true, err
	case "quantum_potential":
		invH2 := float32(0)
		scale := float32(0)

		if len(floatParams) > 0 {
			invH2 = float32(floatParams[0])
		}

		if len(floatParams) > 1 {
			scale = float32(floatParams[1])
		}

		hloText, err := hlo.RenderQuantumPotential(
			moduleName,
			context.InputDTypes[0],
			count[0],
			invH2,
			scale,
		)

		return hloText, true, err
	case "vector_slice_copy":
		if len(intParams) < 2 {
			return "", false, &loweringError{message: "vector slice copy requires offset and length"}
		}

		hloText, err := hlo.RenderVectorSliceCopy(
			moduleName,
			context.InputDTypes[0],
			count[0],
			int(intParams[0]),
			int(intParams[1]),
		)

		return hloText, true, err
	case "markov_mutual_information":
		if len(intParams) < 2 {
			return "", false, &loweringError{message: "markov mutual information requires x and y counts"}
		}

		hloText, err := hlo.RenderMarkovMutualInformation(
			moduleName,
			context.InputDTypes[0],
			int(intParams[0]),
			int(intParams[1]),
		)

		return hloText, true, err
	default:
		return "", false, nil
	}
}

func (builder *Builder) ExecuteVectorSliceCopy(
	bridge *xlaBridge,
	context LoweringContext,
	offset, length int,
	input *DeviceTensor,
	output *DeviceTensor,
) error {
	intParams := []int64{int64(offset), int64(length)}

	programKey, err := builder.ProgramKeyFor("vector_slice_copy", context, nil, intParams)

	if err != nil {
		return err
	}

	executable, err := builder.loadResearchUnaryParamExecutable(
		bridge,
		programKey,
		context,
		nil,
		intParams,
	)

	if err != nil {
		return err
	}

	return builder.recordExecute(bridge.executeUnary(C.XLAExecutableRef(executable.handle), input.bufferRef(), output.bufferRef()))
}

func physicsSpacingInverse(spacing float32, square bool) float32 {
	dxValue := float64(spacing)

	if dxValue <= 0 {
		dxValue = 1.0
	}

	if square {
		return float32(1.0 / (dxValue * dxValue))
	}

	return float32(1.0 / (2 * dxValue))
}

func physicsLaplacian4InverseDenominator(spacing float32) float32 {
	dxValue := float64(spacing)

	if dxValue <= 0 {
		dxValue = 1.0
	}

	denominator := 12 * dxValue * dxValue

	return float32(1.0 / denominator)
}

func physicsQuantumScale() float32 {
	const reducedPlanck = float32(1.0)
	const mass = float32(1.0)

	return -reducedPlanck * reducedPlanck / (2 * mass)
}
