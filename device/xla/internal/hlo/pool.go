package hlo

import (
	"fmt"
	"strings"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

type PoolParams struct {
	KernelH  int64
	KernelW  int64
	StrideH  int64
	StrideW  int64
	PaddingH int64
	PaddingW int64
}

func ResolvePoolParams(
	kernelH, kernelW, strideH, strideW, paddingH, paddingW int64,
	inHeight, inWidth, outHeight, outWidth int64,
) PoolParams {
	if kernelH == 0 && kernelW == 0 {
		kernelH = inHeight / outHeight
		kernelW = inWidth / outWidth
	}

	if strideH == 0 {
		strideH = kernelH
	}

	if strideW == 0 {
		strideW = kernelW
	}

	return PoolParams{
		KernelH:  kernelH,
		KernelW:  kernelW,
		StrideH:  strideH,
		StrideW:  strideW,
		PaddingH: paddingH,
		PaddingW: paddingW,
	}
}

func RenderMaxPool2D(
	moduleName string,
	elementFormat dtype.DType,
	inputShape tensor.Shape,
	outputShape tensor.Shape,
	poolParams PoolParams,
) (string, error) {
	return renderReduceWindowPool(
		moduleName, elementFormat, inputShape, outputShape, poolParams, true,
	)
}

func RenderAvgPool2D(
	moduleName string,
	elementFormat dtype.DType,
	inputShape tensor.Shape,
	outputShape tensor.Shape,
	poolParams PoolParams,
) (string, error) {
	return renderReduceWindowPool(
		moduleName, elementFormat, inputShape, outputShape, poolParams, false,
	)
}

func renderReduceWindowPool(
	moduleName string,
	elementFormat dtype.DType,
	inputShape tensor.Shape,
	outputShape tensor.Shape,
	poolParams PoolParams,
	useMax bool,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	inputLiteral := reductionInputLiteral(elementType, inputShape)
	outputLiteral := reductionInputLiteral(elementType, outputShape)
	entryLayout := fmt.Sprintf("%s->%s", inputLiteral, outputLiteral)

	initValue := "0"
	combineOp := "add"
	computationName := "%add"

	if useMax {
		initValue = "-inf"
		combineOp = "maximum"
		computationName = "%max"
	}

	windowArea := float64(poolParams.KernelH * poolParams.KernelW)
	body := fmt.Sprintf(`  p0 = %s parameter(0)
  init = %s[] constant(%s)
  pooled = %s reduce-window(p0, init),
    window={1,1,%d,%d},
    stride={1,1,%d,%d},
    padding={(0,0),(0,0),(%d,%d),(%d,%d)},
    to_apply=%s
`, inputLiteral, elementType, initValue, outputLiteral,
		poolParams.KernelH, poolParams.KernelW,
		poolParams.StrideH, poolParams.StrideW,
		poolParams.PaddingH, poolParams.PaddingH,
		poolParams.PaddingW, poolParams.PaddingW,
		computationName)

	if !useMax {
		divisor := fmt.Sprintf("%g", windowArea)
		body += fmt.Sprintf(`  scale = %s[] constant(%s)
  scale_b = %s broadcast(scale), dimensions={0,1,2,3}
  ROOT result = %s multiply(pooled, scale_b)
`, elementType, divisor, outputLiteral, outputLiteral)

		return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

%s {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT result = %s[] %s(lhs, rhs)
}

ENTRY main {
%s}
`, moduleName, entryLayout,
			computationName, elementType, elementType, elementType, combineOp,
			body), nil
	}

	body += fmt.Sprintf("  ROOT result = %s copy(pooled)\n", outputLiteral)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

%s {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT result = %s[] %s(lhs, rhs)
}

ENTRY main {
%s}
`, moduleName, entryLayout,
		computationName, elementType, elementType, elementType, combineOp,
		body), nil
}

func RenderAdaptiveMaxPool2D(
	moduleName string,
	elementFormat dtype.DType,
	inputShape tensor.Shape,
	outputShape tensor.Shape,
) (string, error) {
	return renderAdaptivePool(moduleName, elementFormat, inputShape, outputShape, true)
}

func RenderAdaptiveAvgPool2D(
	moduleName string,
	elementFormat dtype.DType,
	inputShape tensor.Shape,
	outputShape tensor.Shape,
) (string, error) {
	return renderAdaptivePool(moduleName, elementFormat, inputShape, outputShape, false)
}

func renderAdaptivePool(
	moduleName string,
	elementFormat dtype.DType,
	inputShape tensor.Shape,
	outputShape tensor.Shape,
	useMax bool,
) (string, error) {
	inputDimensions := inputShape.Dims()
	outputDimensions := outputShape.Dims()

	if len(inputDimensions) != 4 || len(outputDimensions) != 4 {
		return "", fmt.Errorf("adaptive pool requires NCHW rank-4 tensors")
	}

	inHeight := int64(inputDimensions[2])
	inWidth := int64(inputDimensions[3])
	outHeight := int64(outputDimensions[2])
	outWidth := int64(outputDimensions[3])

	if inHeight%outHeight == 0 && inWidth%outWidth == 0 {
		poolParams := PoolParams{
			KernelH: inHeight / outHeight,
			KernelW: inWidth / outWidth,
			StrideH: inHeight / outHeight,
			StrideW: inWidth / outWidth,
		}

		if useMax {
			return RenderMaxPool2D(moduleName, elementFormat, inputShape, outputShape, poolParams)
		}

		return RenderAvgPool2D(moduleName, elementFormat, inputShape, outputShape, poolParams)
	}

	return renderAdaptivePoolExplicit(
		moduleName, elementFormat, inputShape, outputShape, useMax,
	)
}

func renderAdaptivePoolExplicit(
	moduleName string,
	elementFormat dtype.DType,
	inputShape tensor.Shape,
	outputShape tensor.Shape,
	useMax bool,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	inputDimensions := inputShape.Dims()
	outputDimensions := outputShape.Dims()
	inHeight := int64(inputDimensions[2])
	inWidth := int64(inputDimensions[3])
	outHeight := int64(outputDimensions[2])
	outWidth := int64(outputDimensions[3])

	inputLiteral := reductionInputLiteral(elementType, inputShape)
	outputLiteral := reductionInputLiteral(elementType, outputShape)
	entryLayout := fmt.Sprintf("%s->%s", inputLiteral, outputLiteral)

	initValue := "0"
	combineOp := "add"
	computationName := "%add"

	if useMax {
		initValue = "-inf"
		combineOp = "maximum"
		computationName = "%max"
	}

	var bodyLines []string
	bodyLines = append(bodyLines, fmt.Sprintf("  p0 = %s parameter(0)", inputLiteral))
	bodyLines = append(bodyLines, fmt.Sprintf("  zero = %s[] constant(0)", elementType))
	bodyLines = append(bodyLines, fmt.Sprintf("  out = %s broadcast(zero), dimensions={0,1,2,3}", outputLiteral))

	updateIndex := 0

	for outRow := int64(0); outRow < outHeight; outRow++ {
		startRow := (outRow * inHeight) / outHeight
		endRow := ((outRow + 1) * inHeight) / outHeight

		for outCol := int64(0); outCol < outWidth; outCol++ {
			startCol := (outCol * inWidth) / outWidth
			endCol := ((outCol + 1) * inWidth) / outWidth
			sliceName := fmt.Sprintf("slice_%d", updateIndex)
			reduceName := fmt.Sprintf("reduced_%d", updateIndex)
			updateName := fmt.Sprintf("out_%d", updateIndex)

			bodyLines = append(bodyLines, fmt.Sprintf(
				"  %s = %s slice(p0, [0,0,%d,%d], [%d,%d,%d,%d])",
				sliceName, inputLiteral,
				startRow, startCol,
				inputDimensions[0], inputDimensions[1], endRow, endCol,
			))

			bodyLines = append(bodyLines, fmt.Sprintf(
				"  init_%d = %s[] constant(%s)",
				updateIndex, elementType, initValue,
			))

			bodyLines = append(bodyLines, fmt.Sprintf(
				"  %s = %s[] reduce(%s, init_%d), dimensions={2,3}, to_apply=%s",
				reduceName, elementType, sliceName, updateIndex, computationName,
			))

			reduceReshape := fmt.Sprintf("reduced_reshape_%d", updateIndex)
			bodyLines = append(bodyLines, fmt.Sprintf(
				"  %s = %s[%d,%d,1,1]{3,2,1,0} reshape(%s)",
				reduceReshape, elementType, inputDimensions[0], inputDimensions[1], reduceName,
			))

			if !useMax {
				windowCount := (endRow - startRow) * (endCol - startCol)
				bodyLines = append(bodyLines, fmt.Sprintf(
					"  scale_%d = %s[] constant(%g)",
					updateIndex, elementType, 1.0/float64(windowCount),
				))
				bodyLines = append(bodyLines, fmt.Sprintf(
					"  %s = %s multiply(%s, broadcast(scale_%d, dimensions={0,1,2,3}))",
					reduceReshape, reduceReshape, reduceReshape, updateIndex,
				))
			}

			bodyLines = append(bodyLines, fmt.Sprintf(
				"  %s = %s dynamic-update-slice(out, %s, {0,0,%d,%d})",
				updateName, outputLiteral, reduceReshape, outRow, outCol,
			))

			bodyLines = append(bodyLines, fmt.Sprintf("  out = %s copy(%s)", updateName, updateName))
			updateIndex++
		}
	}

	if !useMax {
		bodyLines[len(bodyLines)-1] = fmt.Sprintf("  ROOT result = %s copy(out)", outputLiteral)

		return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

%s {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT result = %s[] %s(lhs, rhs)
}

ENTRY main {
%s
}
`, moduleName, entryLayout,
			computationName, elementType, elementType, elementType, combineOp,
			strings.Join(bodyLines, "\n")), nil
	}

	bodyLines[len(bodyLines)-1] = fmt.Sprintf("  ROOT result = %s copy(out)", outputLiteral)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

%s {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT result = %s[] %s(lhs, rhs)
}

ENTRY main {
%s
}
`, moduleName, entryLayout,
		computationName, elementType, elementType, elementType, combineOp,
		strings.Join(bodyLines, "\n")), nil
}
