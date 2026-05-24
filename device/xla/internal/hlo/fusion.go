package hlo

import (
	"fmt"
	"strings"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func RenderMatmulBiasGelu(
	moduleName string,
	elementFormat dtype.DType,
	leftShape tensor.Shape,
	rightShape tensor.Shape,
	outputShape tensor.Shape,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	leftLiteral := reductionInputLiteral(elementType, leftShape)
	rightLiteral := reductionInputLiteral(elementType, rightShape)
	outputLiteral := reductionInputLiteral(elementType, outputShape)
	biasLiteral := fmt.Sprintf("%s[%d]{0}", elementType, outputShape.Dims()[1])
	entryLayout := fmt.Sprintf("%s,%s,%s->%s", leftLiteral, rightLiteral, biasLiteral, outputLiteral)

	geluBody := renderGeluErfBody(elementType, outputLiteral, "biased", "activated")

	body := strings.Join([]string{
		fmt.Sprintf("  lhs = %s parameter(0)", leftLiteral),
		fmt.Sprintf("  rhs = %s parameter(1)", rightLiteral),
		fmt.Sprintf("  bias = %s parameter(2)", biasLiteral),
		fmt.Sprintf("  mm = %s dot(lhs, rhs), lhs_contracting_dimensions={1}, rhs_contracting_dimensions={0}", outputLiteral),
		fmt.Sprintf("  bias_b = %s broadcast(bias), dimensions={1}", outputLiteral),
		fmt.Sprintf("  biased = %s add(mm, bias_b)", outputLiteral),
		geluBody,
		fmt.Sprintf("  ROOT result = %s activated", outputLiteral),
	}, "\n")

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
%s
}
`, moduleName, entryLayout, body), nil
}

func RenderLayernormResidual(
	moduleName string,
	elementFormat dtype.DType,
	inputShape tensor.Shape,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	if len(inputShape.Dims()) != 2 {
		return "", fmt.Errorf("layernorm residual requires rank-2 input")
	}

	rowCount := inputShape.Dims()[0]
	lastDim := inputShape.Dims()[1]
	inputLiteral := reductionInputLiteral(elementType, inputShape)
	scaleLiteral := fmt.Sprintf("%s[%d]{0}", elementType, lastDim)
	rowLiteral := fmt.Sprintf("%s[%d]{0}", elementType, rowCount)
	entryLayout := fmt.Sprintf("%s,%s,%s,%s->%s",
		inputLiteral, scaleLiteral, scaleLiteral, inputLiteral, inputLiteral)

	body := strings.Join([]string{
		fmt.Sprintf("  input = %s parameter(0)", inputLiteral),
		fmt.Sprintf("  scale = %s parameter(1)", scaleLiteral),
		fmt.Sprintf("  bias = %s parameter(2)", scaleLiteral),
		fmt.Sprintf("  residual = %s parameter(3)", inputLiteral),
		fmt.Sprintf("  zero = %s[] constant(0)", elementType),
		fmt.Sprintf("  eps = %s[] constant(%g)", elementType, layerNormEpsilon),
		fmt.Sprintf("  denom = %s[] constant(%d)", elementType, lastDim),
		fmt.Sprintf("  row_mean = %s reduce(input, zero), dimensions={1}, to_apply=%%add", rowLiteral),
		fmt.Sprintf("  row_mean_b = %s broadcast(row_mean), dimensions={0}", inputLiteral),
		fmt.Sprintf("  centered = %s subtract(input, row_mean_b)", inputLiteral),
		fmt.Sprintf("  sq = %s multiply(centered, centered)", inputLiteral),
		fmt.Sprintf("  row_var = %s reduce(sq, zero), dimensions={1}, to_apply=%%add", rowLiteral),
		fmt.Sprintf("  row_var_b = %s broadcast(row_var), dimensions={0}", inputLiteral),
		fmt.Sprintf("  mean_var = %s divide(row_var_b, broadcast(denom, dimensions={0,1}))", inputLiteral),
		fmt.Sprintf("  inv_std = %s rsqrt(add(mean_var, broadcast(eps, dimensions={0,1})))", inputLiteral),
		fmt.Sprintf("  normalized = %s multiply(centered, inv_std)", inputLiteral),
		fmt.Sprintf("  scale_b = %s broadcast(scale), dimensions={0}", inputLiteral),
		fmt.Sprintf("  bias_b = %s broadcast(bias), dimensions={0}", inputLiteral),
		fmt.Sprintf("  normed = %s add(multiply(normalized, scale_b), bias_b)", inputLiteral),
		fmt.Sprintf("  ROOT result = %s add(normed, residual)", inputLiteral),
	}, "\n")

	return renderNormModule(moduleName, entryLayout, elementType, body), nil
}

func renderGeluErfBody(elementType, shapeLiteral, inputName, outputName string) string {
	return fmt.Sprintf(`  inv_sqrt2 = %s[] constant(%g)
  inv_sqrt2_b = %s broadcast(inv_sqrt2), dimensions={}
  scaled = %s multiply(%s, inv_sqrt2_b)
  erf_val = %s erf(scaled)
  one = %s[] constant(1)
  one_b = %s broadcast(one), dimensions={}
  inner = %s add(one_b, erf_val)
  half = %s[] constant(0.5)
  half_b = %s broadcast(half), dimensions={}
  scaled_inner = %s multiply(half_b, inner)
  %s = %s multiply(%s, scaled_inner)
`, elementType, sqrtTwoInverse, shapeLiteral, shapeLiteral, inputName, shapeLiteral,
		elementType, shapeLiteral, shapeLiteral, elementType, shapeLiteral, shapeLiteral, outputName, shapeLiteral, inputName)
}
