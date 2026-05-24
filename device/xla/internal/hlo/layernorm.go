package hlo

import (
	"fmt"
	"strings"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

const layerNormEpsilon = 1e-5
const rmsNormEpsilon = 1e-6
const normEpsilon = 1e-5

func RenderLayerNorm(
	moduleName string,
	elementFormat dtype.DType,
	inputShape tensor.Shape,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	if len(inputShape.Dims()) != 2 {
		return "", fmt.Errorf("layer norm requires rank-2 input")
	}

	rowCount := inputShape.Dims()[0]
	lastDim := inputShape.Dims()[1]
	inputLiteral := reductionInputLiteral(elementType, inputShape)
	scaleLiteral := fmt.Sprintf("%s[%d]{0}", elementType, lastDim)
	rowLiteral := fmt.Sprintf("%s[%d]{0}", elementType, rowCount)
	entryLayout := fmt.Sprintf("%s,%s,%s->%s", inputLiteral, scaleLiteral, scaleLiteral, inputLiteral)

	body := strings.Join([]string{
		fmt.Sprintf("  input = %s parameter(0)", inputLiteral),
		fmt.Sprintf("  scale = %s parameter(1)", scaleLiteral),
		fmt.Sprintf("  bias = %s parameter(2)", scaleLiteral),
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
		fmt.Sprintf("  ROOT result = %s add(multiply(normalized, scale_b), bias_b)", inputLiteral),
	}, "\n")

	return renderNormModule(moduleName, entryLayout, elementType, body), nil
}

func RenderRMSNorm(
	moduleName string,
	elementFormat dtype.DType,
	inputShape tensor.Shape,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	if len(inputShape.Dims()) != 2 {
		return "", fmt.Errorf("rms norm requires rank-2 input")
	}

	rowCount := inputShape.Dims()[0]
	lastDim := inputShape.Dims()[1]
	inputLiteral := reductionInputLiteral(elementType, inputShape)
	scaleLiteral := fmt.Sprintf("%s[%d]{0}", elementType, lastDim)
	rowLiteral := fmt.Sprintf("%s[%d]{0}", elementType, rowCount)
	entryLayout := fmt.Sprintf("%s,%s->%s", inputLiteral, scaleLiteral, inputLiteral)

	body := strings.Join([]string{
		fmt.Sprintf("  input = %s parameter(0)", inputLiteral),
		fmt.Sprintf("  scale = %s parameter(1)", scaleLiteral),
		fmt.Sprintf("  zero = %s[] constant(0)", elementType),
		fmt.Sprintf("  eps = %s[] constant(%g)", elementType, rmsNormEpsilon),
		fmt.Sprintf("  denom = %s[] constant(%d)", elementType, lastDim),
		fmt.Sprintf("  sq = %s multiply(input, input)", inputLiteral),
		fmt.Sprintf("  row_var = %s reduce(sq, zero), dimensions={1}, to_apply=%%add", rowLiteral),
		fmt.Sprintf("  row_var_b = %s broadcast(row_var), dimensions={0}", inputLiteral),
		fmt.Sprintf("  mean_sq = %s divide(row_var_b, broadcast(denom, dimensions={0,1}))", inputLiteral),
		fmt.Sprintf("  inv_rms = %s rsqrt(add(mean_sq, broadcast(eps, dimensions={0,1})))", inputLiteral),
		fmt.Sprintf("  normalized = %s multiply(input, inv_rms)", inputLiteral),
		fmt.Sprintf("  scale_b = %s broadcast(scale), dimensions={0}", inputLiteral),
		fmt.Sprintf("  ROOT result = %s multiply(normalized, scale_b)", inputLiteral),
	}, "\n")

	return renderNormModule(moduleName, entryLayout, elementType, body), nil
}

func RenderBatchNormEval(
	moduleName string,
	elementFormat dtype.DType,
	inputShape tensor.Shape,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	dimensions := inputShape.Dims()
	channels := dimensions[1]
	inputLiteral := reductionInputLiteral(elementType, inputShape)
	channelLiteral := fmt.Sprintf("%s[%d]{0}", elementType, channels)
	entryLayout := fmt.Sprintf("%s,%s,%s,%s,%s->%s",
		inputLiteral, channelLiteral, channelLiteral, channelLiteral, channelLiteral, inputLiteral)

	body := strings.Join([]string{
		fmt.Sprintf("  input = %s parameter(0)", inputLiteral),
		fmt.Sprintf("  scale = %s parameter(1)", channelLiteral),
		fmt.Sprintf("  bias = %s parameter(2)", channelLiteral),
		fmt.Sprintf("  mean = %s parameter(3)", channelLiteral),
		fmt.Sprintf("  variance = %s parameter(4)", channelLiteral),
		fmt.Sprintf("  eps = %s[] constant(%g)", elementType, normEpsilon),
		fmt.Sprintf("  mean_b = %s broadcast(mean), dimensions={0,2}", inputLiteral),
		fmt.Sprintf("  centered = %s subtract(input, mean_b)", inputLiteral),
		fmt.Sprintf("  var_b = %s broadcast(variance), dimensions={0,2}", inputLiteral),
		fmt.Sprintf("  inv_std = %s rsqrt(add(var_b, broadcast(eps, dimensions={0,2})))", inputLiteral),
		fmt.Sprintf("  normalized = %s multiply(centered, inv_std)", inputLiteral),
		fmt.Sprintf("  scale_b = %s broadcast(scale), dimensions={0,2}", inputLiteral),
		fmt.Sprintf("  bias_b = %s broadcast(bias), dimensions={0,2}", inputLiteral),
		fmt.Sprintf("  ROOT result = %s add(multiply(normalized, scale_b), bias_b)", inputLiteral),
	}, "\n")

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
%s
}
`, moduleName, entryLayout, body), nil
}

func RenderInstanceNorm(
	moduleName string,
	elementFormat dtype.DType,
	inputShape tensor.Shape,
) (string, error) {
	return renderChannelStatNorm(moduleName, elementFormat, inputShape)
}

func RenderGroupNorm(
	moduleName string,
	elementFormat dtype.DType,
	inputShape tensor.Shape,
	groups int,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	dimensions := inputShape.Dims()

	if len(dimensions) != 3 {
		return "", fmt.Errorf("group norm requires rank-3 input")
	}

	batchSize := dimensions[0]
	channels := dimensions[1]
	spatial := dimensions[2]

	if channels%groups != 0 {
		return "", fmt.Errorf("group norm requires channels divisible by groups")
	}

	channelsPerGroup := channels / groups
	inputLiteral := reductionInputLiteral(elementType, inputShape)
	scaleLiteral := fmt.Sprintf("%s[%d]{0}", elementType, channels)
	reshapedLiteral := fmt.Sprintf("%s[%d,%d,%d,%d]{3,2,1,0}", elementType, batchSize, groups, channelsPerGroup, spatial)
	groupLiteral := fmt.Sprintf("%s[%d,%d]{1,0}", elementType, batchSize, groups)
	scaleReshapeLiteral := fmt.Sprintf("%s[1,%d,%d,1]{3,2,1,0}", elementType, groups, channelsPerGroup)
	entryLayout := fmt.Sprintf("%s,%s,%s->%s", inputLiteral, scaleLiteral, scaleLiteral, inputLiteral)

	body := strings.Join([]string{
		fmt.Sprintf("  input = %s parameter(0)", inputLiteral),
		fmt.Sprintf("  scale = %s parameter(1)", scaleLiteral),
		fmt.Sprintf("  bias = %s parameter(2)", scaleLiteral),
		fmt.Sprintf("  zero = %s[] constant(0)", elementType),
		fmt.Sprintf("  eps = %s[] constant(%g)", elementType, normEpsilon),
		fmt.Sprintf("  group_elems = %s[] constant(%d)", elementType, channelsPerGroup*spatial),
		fmt.Sprintf("  reshaped = %s reshape(input)", reshapedLiteral),
		fmt.Sprintf("  group_mean = %s reduce(reshaped, zero), dimensions={2,3}, to_apply=%%add", groupLiteral),
		fmt.Sprintf("  group_mean_b = %s broadcast(group_mean), dimensions={1,2,3}", reshapedLiteral),
		fmt.Sprintf("  centered = %s subtract(reshaped, group_mean_b)", reshapedLiteral),
		fmt.Sprintf("  sq = %s multiply(centered, centered)", reshapedLiteral),
		fmt.Sprintf("  group_var = %s reduce(sq, zero), dimensions={2,3}, to_apply=%%add", groupLiteral),
		fmt.Sprintf("  group_var_b = %s broadcast(group_var), dimensions={1,2,3}", reshapedLiteral),
		fmt.Sprintf("  mean_var = %s divide(group_var_b, broadcast(group_elems, dimensions={1,2,3}))", reshapedLiteral),
		fmt.Sprintf("  inv_std = %s rsqrt(add(mean_var, broadcast(eps, dimensions={1,2,3})))", reshapedLiteral),
		fmt.Sprintf("  normalized = %s multiply(centered, inv_std)", reshapedLiteral),
		fmt.Sprintf("  scale_r = %s reshape(scale)", scaleReshapeLiteral),
		fmt.Sprintf("  scale_b = %s broadcast(scale_r), dimensions={0,3}", reshapedLiteral),
		fmt.Sprintf("  bias_r = %s reshape(bias)", scaleReshapeLiteral),
		fmt.Sprintf("  bias_b = %s broadcast(bias_r), dimensions={0,3}", reshapedLiteral),
		fmt.Sprintf("  scaled = %s multiply(normalized, scale_b)", reshapedLiteral),
		fmt.Sprintf("  shifted = %s add(scaled, bias_b)", reshapedLiteral),
		fmt.Sprintf("  ROOT result = %s reshape(shifted)", inputLiteral),
	}, "\n")

	return renderNormModule(moduleName, entryLayout, elementType, body), nil
}

func renderChannelStatNorm(
	moduleName string,
	elementFormat dtype.DType,
	inputShape tensor.Shape,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	dimensions := inputShape.Dims()
	channels := dimensions[1]
	inputLiteral := reductionInputLiteral(elementType, inputShape)
	scaleLiteral := fmt.Sprintf("%s[%d]{0}", elementType, channels)
	statLiteral := fmt.Sprintf("%s[%d,%d]{1,0}", elementType, dimensions[0], channels)
	entryLayout := fmt.Sprintf("%s,%s,%s->%s", inputLiteral, scaleLiteral, scaleLiteral, inputLiteral)

	body := strings.Join([]string{
		fmt.Sprintf("  input = %s parameter(0)", inputLiteral),
		fmt.Sprintf("  scale = %s parameter(1)", scaleLiteral),
		fmt.Sprintf("  bias = %s parameter(2)", scaleLiteral),
		fmt.Sprintf("  zero = %s[] constant(0)", elementType),
		fmt.Sprintf("  eps = %s[] constant(%g)", elementType, normEpsilon),
		fmt.Sprintf("  spatial_count = %s[] constant(%d)", elementType, dimensions[2]),
		fmt.Sprintf("  channel_mean = %s reduce(input, zero), dimensions={2}, to_apply=%%add", statLiteral),
		fmt.Sprintf("  channel_mean_b = %s broadcast(channel_mean), dimensions={0,2}", inputLiteral),
		fmt.Sprintf("  centered = %s subtract(input, channel_mean_b)", inputLiteral),
		fmt.Sprintf("  sq = %s multiply(centered, centered)", inputLiteral),
		fmt.Sprintf("  channel_var = %s reduce(sq, zero), dimensions={2}, to_apply=%%add", statLiteral),
		fmt.Sprintf("  channel_var_b = %s broadcast(channel_var), dimensions={0,2}", inputLiteral),
		fmt.Sprintf("  mean_var = %s divide(channel_var_b, broadcast(spatial_count, dimensions={0,2}))", inputLiteral),
		fmt.Sprintf("  inv_std = %s rsqrt(add(mean_var, broadcast(eps, dimensions={0,2})))", inputLiteral),
		fmt.Sprintf("  normalized = %s multiply(centered, inv_std)", inputLiteral),
		fmt.Sprintf("  scale_b = %s broadcast(scale), dimensions={0,2}", inputLiteral),
		fmt.Sprintf("  bias_b = %s broadcast(bias), dimensions={0,2}", inputLiteral),
		fmt.Sprintf("  ROOT result = %s add(multiply(normalized, scale_b), bias_b)", inputLiteral),
	}, "\n")

	return renderNormModule(moduleName, entryLayout, elementType, body), nil
}

func renderNormModule(
	moduleName string,
	entryLayout string,
	elementType string,
	body string,
) string {
	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

%%add {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT result = %s[] add(lhs, rhs)
}

ENTRY main {
%s
}
`, moduleName, entryLayout, elementType, elementType, elementType, body)
}
