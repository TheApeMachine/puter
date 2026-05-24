package hlo

import (
	"fmt"
	"strings"

	"github.com/theapemachine/manifesto/dtype"
)

func RenderCATE(moduleName string, elementFormat dtype.DType, count int) (string, error) {
	return renderBinaryVectorOp(moduleName, elementFormat, count, "subtract")
}

func RenderCounterfactual(
	moduleName string,
	elementFormat dtype.DType,
	count int,
	slope float32,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	vectorLiteral := fmt.Sprintf("%s[%d]{0}", elementType, count)
	entryLayout := fmt.Sprintf("%s,%s,%s->%s", vectorLiteral, vectorLiteral, vectorLiteral, vectorLiteral)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
  observed_y = %s parameter(0)
  observed_x = %s parameter(1)
  counterfactual_x = %s parameter(2)
  delta_x = %s subtract(counterfactual_x, observed_x)
  slope_c = %s[] constant(%g)
  slope_b = %s broadcast(slope_c), dimensions={0}
  scaled = %s multiply(delta_x, slope_b)
  ROOT result = %s add(observed_y, scaled)
}
`, moduleName, entryLayout,
		vectorLiteral, vectorLiteral, vectorLiteral,
		vectorLiteral, elementType, slope, vectorLiteral, vectorLiteral, vectorLiteral), nil
}

func RenderBackdoorAdjustment(
	moduleName string,
	elementFormat dtype.DType,
	xCount, zCount, yCount int,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	conditionalLiteral := fmt.Sprintf("%s[%d,%d,%d]{2,1,0}", elementType, xCount, zCount, yCount)
	marginalLiteral := fmt.Sprintf("%s[%d]{0}", elementType, zCount)
	outputLiteral := fmt.Sprintf("%s[%d,%d]{1,0}", elementType, xCount, yCount)
	entryLayout := fmt.Sprintf("%s,%s->%s", conditionalLiteral, marginalLiteral, outputLiteral)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

%%add {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT result = %s[] add(lhs, rhs)
}

ENTRY main {
  conditional = %s parameter(0)
  marginal = %s parameter(1)
  marginal_b = %s[%d,%d,1]{2,1,0} broadcast(marginal), dimensions={1,0}
  weighted = %s multiply(conditional, marginal_b)
  zero = %s[] constant(0)
  ROOT result = %s reduce(weighted, zero), dimensions={1}, to_apply=%%add
}
`, moduleName, entryLayout,
		elementType, elementType, elementType,
		conditionalLiteral, marginalLiteral,
		elementType, xCount, yCount, conditionalLiteral,
		elementType, outputLiteral), nil
}

func RenderDoIntervene(
	moduleName string,
	elementFormat dtype.DType,
	nodeCount, intervenedCount int,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	adjacencyLiteral := fmt.Sprintf("%s[%d,%d]{1,0}", elementType, nodeCount, nodeCount)
	intervenedLiteral := fmt.Sprintf("s32[%d]{0}", intervenedCount)
	entryLayout := fmt.Sprintf("%s,%s->%s", adjacencyLiteral, intervenedLiteral, adjacencyLiteral)

	if intervenedCount == 0 {
		return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
  adjacency = %s parameter(0)
  _intervened = %s parameter(1)
  ROOT result = %s copy(adjacency)
}
`, moduleName, entryLayout, adjacencyLiteral, intervenedLiteral, adjacencyLiteral), nil
	}

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

%%or {
  lhs = pred[] parameter(0)
  rhs = pred[] parameter(1)
  ROOT result = pred[] or(lhs, rhs)
}

ENTRY main {
  adjacency = %s parameter(0)
  intervened = %s parameter(1)
  col_idx = s32[%d,%d]{1,0} reshape(iota(s32[%d]{0}), dimensions={1,%d})
  int_row = s32[1,%d]{1,0} reshape(intervened), dimensions={0,%d}
  match = pred[%d,%d]{1,0} compare(col_idx, int_row), direction=EQ
  zero_mask = pred[%d,%d]{1,0} reduce(match, pred[] constant(false)), dimensions={1}, to_apply=%%or
  zero = %s[] constant(0)
  zero_b = %s broadcast(zero), dimensions={0,1}
  ROOT result = %s select(zero_mask, zero_b, adjacency)
}
`, moduleName, entryLayout,
		adjacencyLiteral, intervenedLiteral,
		nodeCount, nodeCount, nodeCount, nodeCount,
		intervenedCount, intervenedCount,
		nodeCount, intervenedCount,
		nodeCount, nodeCount,
		elementType, adjacencyLiteral, adjacencyLiteral), nil
}

func RenderDAGMarkovFactorization(
	moduleName string,
	elementFormat dtype.DType,
	count int,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	vectorLiteral := fmt.Sprintf("%s[%d]{0}", elementType, count)
	scalarLiteral := fmt.Sprintf("%s[]", elementType)
	entryLayout := fmt.Sprintf("%s->%s", vectorLiteral, scalarLiteral)
	epsilon := float32(1e-12)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

%%multiply {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT result = %s[] multiply(lhs, rhs)
}

%%max {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT result = %s[] maximum(lhs, rhs)
}

ENTRY main {
  conditionals = %s parameter(0)
  eps = %s[] constant(%g)
  clamped = %s maximum(conditionals, eps)
  one = %s[] constant(1)
  ROOT result = %s[] reduce(clamped, one), dimensions={0}, to_apply=%%multiply
}
`, moduleName, entryLayout,
		elementType, elementType, elementType,
		elementType, elementType, elementType,
		vectorLiteral, elementType, epsilon, vectorLiteral, elementType, scalarLiteral), nil
}

func RenderIVEstimate(
	moduleName string,
	elementFormat dtype.DType,
	count int,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	vectorLiteral := fmt.Sprintf("%s[%d]{0}", elementType, count)
	scalarLiteral := fmt.Sprintf("%s[]", elementType)
	entryLayout := fmt.Sprintf("%s,%s,%s->%s", vectorLiteral, vectorLiteral, vectorLiteral, scalarLiteral)
	countFloat := float32(count)
	epsilon := float32(1e-12)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

%%add {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT result = %s[] add(lhs, rhs)
}

%%multiply {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT result = %s[] multiply(lhs, rhs)
}

ENTRY main {
  instrument = %s parameter(0)
  treatment = %s parameter(1)
  outcome = %s parameter(2)
  inv_n = %s[] constant(%g)
  mean_z = %s multiply(%s reduce(instrument, %s[] constant(0)), dimensions={0}, to_apply=%%add), inv_n)
  mean_x = %s multiply(%s reduce(treatment, %s[] constant(0)), dimensions={0}, to_apply=%%add), inv_n)
  mean_y = %s multiply(%s reduce(outcome, %s[] constant(0)), dimensions={0}, to_apply=%%add), inv_n)
  mean_z_b = %s broadcast(mean_z), dimensions={0}
  mean_x_b = %s broadcast(mean_x), dimensions={0}
  mean_y_b = %s broadcast(mean_y), dimensions={0}
  delta_z = %s subtract(instrument, mean_z_b)
  delta_x = %s subtract(treatment, mean_x_b)
  delta_y = %s subtract(outcome, mean_y_b)
  zy = %s multiply(delta_z, delta_y)
  zx = %s multiply(delta_z, delta_x)
  zero = %s[] constant(0)
  cov_zy = %s reduce(zy, zero), dimensions={0}, to_apply=%%add
  cov_zx = %s reduce(zx, zero), dimensions={0}, to_apply=%%add
  pos = pred[] compare(cov_zx, zero), direction=GE
  neg = pred[] compare(cov_zx, zero), direction=LE
  neg_eps = %s[] constant(%g)
  pos_safe = %s maximum(cov_zx, %s[] constant(%g))
  neg_safe = %s minimum(cov_zx, neg_eps)
  signed_safe = %s select(pos, pos_safe, %s select(neg, neg_safe, %s[] constant(%g)))
  ROOT result = %s divide(cov_zy, signed_safe)
}
`, moduleName, entryLayout,
		elementType, elementType, elementType,
		elementType, elementType, elementType,
		vectorLiteral, vectorLiteral, vectorLiteral,
		elementType, countFloat, elementType, elementType, elementType,
		elementType, elementType, elementType,
		elementType, elementType, elementType,
		vectorLiteral, vectorLiteral, vectorLiteral,
		vectorLiteral, vectorLiteral, vectorLiteral,
		vectorLiteral, vectorLiteral,
		elementType, elementType, elementType,
		elementType, epsilon, elementType, epsilon,
		elementType, elementType, elementType, epsilon, scalarLiteral), nil
}

func RenderMarkovFlow(
	moduleName string,
	elementFormat dtype.DType,
	nodeCount int,
	targetLabel int32,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	miLiteral := fmt.Sprintf("%s[%d,%d]{1,0}", elementType, nodeCount, nodeCount)
	partitionLiteral := fmt.Sprintf("s32[%d]{0}", nodeCount)
	outputLiteral := fmt.Sprintf("%s[%d]{0}", elementType, nodeCount)
	entryLayout := fmt.Sprintf("%s,%s->%s", miLiteral, partitionLiteral, outputLiteral)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

%%add {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT result = %s[] add(lhs, rhs)
}

ENTRY main {
  mutual_information = %s parameter(0)
  partition = %s parameter(1)
  target = s32[] constant(%d)
  target_b = s32[%d]{0} broadcast(target), dimensions={0}
  row_match = pred[%d]{0} compare(partition, target_b), direction=EQ
  zero_label = s32[] constant(0)
  zero_b = s32[%d]{0} broadcast(zero_label), dimensions={0}
  col_match = pred[%d]{0} compare(partition, zero_b), direction=EQ
  col_match_row = pred[%d,%d]{1,0} broadcast(col_match), dimensions={1,0}
  masked = %s select(col_match_row, mutual_information, %s broadcast(%s[] constant(0)), dimensions={0,1})
  zero = %s[] constant(0)
  flow = %s[%d]{0} reduce(masked, zero), dimensions={1}, to_apply=%%add
  zero_out = %s[] constant(0)
  zero_out_b = %s broadcast(zero_out), dimensions={0}
  ROOT result = %s select(row_match, flow, zero_out_b)
}
`, moduleName, entryLayout,
		elementType, elementType, elementType,
		miLiteral, partitionLiteral, targetLabel, nodeCount,
		nodeCount, nodeCount, nodeCount, nodeCount,
		miLiteral, miLiteral, elementType,
		elementType, elementType, nodeCount,
		elementType, elementType, outputLiteral), nil
}

func RenderFrontdoorAdjustment(
	moduleName string,
	elementFormat dtype.DType,
	xCount, mCount, yCount int,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	mediatorLiteral := fmt.Sprintf("%s[%d,%d]{1,0}", elementType, xCount, mCount)
	outcomeLiteral := fmt.Sprintf("%s[%d,%d,%d]{2,1,0}", elementType, xCount, mCount, yCount)
	marginalLiteral := fmt.Sprintf("%s[%d]{0}", elementType, xCount)
	outputLiteral := fmt.Sprintf("%s[%d,%d]{1,0}", elementType, xCount, yCount)
	entryLayout := fmt.Sprintf("%s,%s,%s->%s", mediatorLiteral, outcomeLiteral, marginalLiteral, outputLiteral)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

%%add {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT result = %s[] add(lhs, rhs)
}

%%multiply {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT result = %s[] multiply(lhs, rhs)
}

ENTRY main {
  mediator_given_x = %s parameter(0)
  outcome_given_xm = %s parameter(1)
  marginal_x = %s parameter(2)
  marginal_b = %s[%d,1,1]{2,1,0} broadcast(marginal_x), dimensions={0,1,2}
  weighted_outcome = %s multiply(outcome_given_xm, marginal_b)
  zero = %s[] constant(0)
  summed_xprime = %s[%d,%d,1]{2,1,0} reduce(weighted_outcome, zero), dimensions={0}, to_apply=%%add
  pmx = %s[%d,%d,1]{2,1,0} reshape(mediator_given_x), dimensions={%d,%d,1}
  inner = %s multiply(pmx, summed_xprime)
  ROOT result = %s[%d,%d]{1,0} reduce(inner, zero), dimensions={1}, to_apply=%%add
}
`, moduleName, entryLayout,
		elementType, elementType, elementType,
		elementType, elementType, elementType,
		mediatorLiteral, outcomeLiteral, marginalLiteral,
		elementType, xCount, outcomeLiteral,
		elementType, xCount, yCount,
		elementType, xCount, mCount, xCount, mCount,
		outcomeLiteral, outputLiteral, xCount, yCount), nil
}

func RenderCholesky(
	moduleName string,
	elementFormat dtype.DType,
	matrixOrder int,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	if matrixOrder <= 0 {
		return "", fmt.Errorf("cholesky requires positive matrix order")
	}

	matrixLiteral := fmt.Sprintf("%s[%d,%d]{1,0}", elementType, matrixOrder, matrixOrder)
	entryLayout := fmt.Sprintf("%s->%s", matrixLiteral, matrixLiteral)

	var body strings.Builder

	fmt.Fprintf(&body, "HloModule %s, entry_computation_layout={%s}\n\n", moduleName, entryLayout)
	fmt.Fprintf(&body, "ENTRY main {\n")
	fmt.Fprintf(&body, "  input = %s parameter(0)\n", matrixLiteral)
	fmt.Fprintf(&body, "  zero = %s[] constant(0)\n", elementType)
	fmt.Fprintf(&body, "  zero_b = %s broadcast(zero), dimensions={0,1}\n", matrixLiteral)
	fmt.Fprintf(&body, "  current = %s multiply(zero_b, zero_b)\n", matrixLiteral)

	for rowIndex := 0; rowIndex < matrixOrder; rowIndex++ {
		for colIndex := 0; colIndex <= rowIndex; colIndex++ {
			sumName := fmt.Sprintf("sum_%d_%d", rowIndex, colIndex)
			fmt.Fprintf(&body, "  %s = %s dynamic-slice(input, s32[] constant(%d), s32[] constant(%d)), dynamic_slice_sizes={1,1}\n",
				sumName, elementType, rowIndex, colIndex)

			for innerIndex := 0; innerIndex < colIndex; innerIndex++ {
				leftName := fmt.Sprintf("l_%d_%d_%d", rowIndex, colIndex, innerIndex)
				rightName := fmt.Sprintf("r_%d_%d_%d", rowIndex, colIndex, innerIndex)
				prodName := fmt.Sprintf("p_%d_%d_%d", rowIndex, colIndex, innerIndex)
				nextSum := fmt.Sprintf("s_%d_%d_%d", rowIndex, colIndex, innerIndex)
				fmt.Fprintf(&body, "  %s = %s dynamic-slice(current, s32[] constant(%d), s32[] constant(%d)), dynamic_slice_sizes={1,1}\n",
					leftName, elementType, rowIndex, innerIndex)
				fmt.Fprintf(&body, "  %s = %s dynamic-slice(current, s32[] constant(%d), s32[] constant(%d)), dynamic_slice_sizes={1,1}\n",
					rightName, elementType, colIndex, innerIndex)
				fmt.Fprintf(&body, "  %s = %s multiply(%s, %s)\n", prodName, elementType, leftName, rightName)
				fmt.Fprintf(&body, "  %s = %s subtract(%s, %s)\n", nextSum, elementType, sumName, prodName)
				sumName = nextSum
			}

			valueName := fmt.Sprintf("val_%d_%d", rowIndex, colIndex)

			if rowIndex == colIndex {
				fmt.Fprintf(&body, "  %s = %s sqrt(%s)\n", valueName, elementType, sumName)
			} else {
				diagName := fmt.Sprintf("diag_%d_%d", rowIndex, colIndex)
				fmt.Fprintf(&body, "  %s = %s dynamic-slice(current, s32[] constant(%d), s32[] constant(%d)), dynamic_slice_sizes={1,1}\n",
					diagName, elementType, colIndex, colIndex)
				fmt.Fprintf(&body, "  %s = %s divide(%s, %s)\n", valueName, elementType, sumName, diagName)
			}

			fmt.Fprintf(&body, "  current = %s dynamic-update-slice(current, %s reshape(%s), s32[] constant(%d), s32[] constant(%d))\n",
				matrixLiteral, matrixLiteral, valueName, rowIndex, colIndex)
		}
	}

	fmt.Fprintf(&body, "  ROOT result = %s copy(current)\n", matrixLiteral)
	fmt.Fprintf(&body, "}\n")

	return body.String(), nil
}

func renderBinaryVectorOp(
	moduleName string,
	elementFormat dtype.DType,
	count int,
	opName string,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	vectorLiteral := fmt.Sprintf("%s[%d]{0}", elementType, count)
	entryLayout := fmt.Sprintf("%s,%s->%s", vectorLiteral, vectorLiteral, vectorLiteral)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
  lhs = %s parameter(0)
  rhs = %s parameter(1)
  ROOT result = %s %s(lhs, rhs)
}
`, moduleName, entryLayout, vectorLiteral, vectorLiteral, vectorLiteral, opName), nil
}
