package hlo

import (
	"fmt"
	"strings"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

type ConvParams struct {
	Strides     []int64
	Paddings    []int64
	Dilations   []int64
	LhsDilation []int64
	Transpose   bool
}

func ConvParamsFromIntParams(intParams []int64, spatialRank int, transpose bool) (ConvParams, error) {
	expected := spatialRank * 3

	if len(intParams) < expected {
		return ConvParams{}, fmt.Errorf("convolution requires %d int params, got %d", expected, len(intParams))
	}

	strides := make([]int64, spatialRank)
	paddings := make([]int64, spatialRank)
	dilations := make([]int64, spatialRank)
	lhsDilation := make([]int64, spatialRank+2)

	for index := range spatialRank + 2 {
		lhsDilation[index] = 1
	}

	for spatialIndex := range spatialRank {
		strides[spatialIndex] = intParams[spatialIndex]
		paddings[spatialIndex] = intParams[spatialRank+spatialIndex]
		dilations[spatialIndex] = intParams[2*spatialRank+spatialIndex]
	}

	if transpose {
		for spatialIndex := range spatialRank {
			lhsDilation[spatialIndex+2] = strides[spatialIndex]
		}
	}

	return ConvParams{
		Strides:     strides,
		Paddings:    paddings,
		Dilations:   dilations,
		LhsDilation: lhsDilation,
		Transpose:   transpose,
	}, nil
}

func ApplyTransposePadding(
	convParams *ConvParams,
	weightShape tensor.Shape,
) error {
	if !convParams.Transpose {
		return nil
	}

	weightDimensions := weightShape.Dims()
	spatialRank := len(convParams.Paddings)

	if len(weightDimensions) != spatialRank+2 {
		return fmt.Errorf("transpose convolution weight rank mismatch")
	}

	for spatialIndex := range spatialRank {
		kernelSize := int64(weightDimensions[spatialIndex+2])
		transposePad := (kernelSize-1)*convParams.Dilations[spatialIndex] - convParams.Paddings[spatialIndex]

		if transposePad < 0 {
			transposePad = 0
		}

		convParams.Paddings[spatialIndex] = transposePad
	}

	return nil
}

func RenderConv2D(
	moduleName string,
	elementFormat dtype.DType,
	inputShape tensor.Shape,
	weightShape tensor.Shape,
	biasShape tensor.Shape,
	outputShape tensor.Shape,
	convParams ConvParams,
) (string, error) {
	return renderConvolution(
		moduleName, elementFormat,
		inputShape, weightShape, biasShape, outputShape,
		convParams,
		"bf01_oi01->bf01",
		2,
	)
}

func RenderConvTranspose2D(
	moduleName string,
	elementFormat dtype.DType,
	inputShape tensor.Shape,
	weightShape tensor.Shape,
	biasShape tensor.Shape,
	outputShape tensor.Shape,
	convParams ConvParams,
) (string, error) {
	params := convParams

	if err := ApplyTransposePadding(&params, weightShape); err != nil {
		return "", err
	}

	return renderConvolution(
		moduleName, elementFormat,
		inputShape, weightShape, biasShape, outputShape,
		params,
		"bf01_io01->bf01",
		2,
	)
}

func RenderConv1D(
	moduleName string,
	elementFormat dtype.DType,
	inputShape tensor.Shape,
	weightShape tensor.Shape,
	biasShape tensor.Shape,
	outputShape tensor.Shape,
	convParams ConvParams,
) (string, error) {
	return renderConvolution(
		moduleName, elementFormat,
		inputShape, weightShape, biasShape, outputShape,
		convParams,
		"bf0_oi0->bf0",
		1,
	)
}

func RenderConv3D(
	moduleName string,
	elementFormat dtype.DType,
	inputShape tensor.Shape,
	weightShape tensor.Shape,
	biasShape tensor.Shape,
	outputShape tensor.Shape,
	convParams ConvParams,
) (string, error) {
	return renderConvolution(
		moduleName, elementFormat,
		inputShape, weightShape, biasShape, outputShape,
		convParams,
		"bf012_oi012->bf012",
		3,
	)
}

func renderConvolution(
	moduleName string,
	elementFormat dtype.DType,
	inputShape tensor.Shape,
	weightShape tensor.Shape,
	biasShape tensor.Shape,
	outputShape tensor.Shape,
	convParams ConvParams,
	dimensionLabels string,
	spatialRank int,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	inputLiteral := reductionInputLiteral(elementType, inputShape)
	weightLiteral := reductionInputLiteral(elementType, weightShape)
	biasLiteral := reductionInputLiteral(elementType, biasShape)
	outputLiteral := reductionInputLiteral(elementType, outputShape)
	entryLayout := fmt.Sprintf(
		"%s,%s,%s->%s",
		inputLiteral, weightLiteral, biasLiteral, outputLiteral,
	)

	strideText := formatConvWindowStride(convParams.Strides)
	paddingText := formatConvPadding(spatialRank, convParams.Paddings)
	lhsDilationText := formatConvDilation(convParams.LhsDilation)
	rhsDilationText := formatConvSpatialDilation(spatialRank, convParams.Dilations)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
  input = %s parameter(0)
  weight = %s parameter(1)
  bias = %s parameter(2)
  convolved = %s convolution(input, weight),
    window={size=1 stride=%s pad=0_0},
    padding=%s,
    lhs_dilation=%s,
    rhs_dilation=%s,
    dim_labels=%s,
    feature_group_count=1
  bias_b = %s broadcast(bias), dimensions={%s}
  ROOT result = %s add(convolved, bias_b)
}
`, moduleName, entryLayout,
		inputLiteral, weightLiteral, biasLiteral,
		outputLiteral, strideText, paddingText,
		lhsDilationText, rhsDilationText, dimensionLabels,
		outputLiteral, broadcastSpatialDimensions(spatialRank), outputLiteral), nil
}

func formatConvWindowStride(strides []int64) string {
	if len(strides) == 1 {
		return fmt.Sprintf("%d", strides[0])
	}

	parts := make([]string, len(strides))

	for index, stride := range strides {
		parts[index] = fmt.Sprintf("%d", stride)
	}

	return strings.Join(parts, "x")
}

func formatConvPadding(spatialRank int, paddings []int64) string {
	pairs := []string{"(0,0)", "(0,0)"}

	for _, padding := range paddings {
		pairs = append(pairs, fmt.Sprintf("(%d,%d)", padding, padding))
	}

	return "{" + strings.Join(pairs, ",") + "}"
}

func formatConvDilation(dilations []int64) string {
	parts := make([]string, len(dilations))

	for index, dilation := range dilations {
		parts[index] = fmt.Sprintf("%d", dilation)
	}

	return "{" + strings.Join(parts, ",") + "}"
}

func formatConvSpatialDilation(spatialRank int, dilations []int64) string {
	prefix := make([]string, 2)

	for index := range prefix {
		prefix[index] = "1"
	}

	parts := append(prefix, make([]string, spatialRank)...)

	for index, dilation := range dilations {
		parts[index+2] = fmt.Sprintf("%d", dilation)
	}

	return "{" + strings.Join(parts, ",") + "}"
}

func broadcastSpatialDimensions(spatialRank int) string {
	dimensions := []string{"0"}

	for dimension := 2; dimension < spatialRank+2; dimension++ {
		dimensions = append(dimensions, fmt.Sprintf("%d", dimension))
	}

	return strings.Join(dimensions, ",")
}

func ConvOutputSize(inputSize, kernelSize, padding, stride, dilation int) int {
	numerator := inputSize + 2*padding - dilation*(kernelSize-1) - 1

	if numerator < 0 {
		return 0
	}

	return numerator/stride + 1
}
