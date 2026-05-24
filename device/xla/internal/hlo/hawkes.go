package hlo

import (
	"fmt"

	"github.com/theapemachine/manifesto/dtype"
)

func RenderHawkesIntensity(
	moduleName string,
	elementFormat dtype.DType,
	eventCount, queryCount int,
	mu, alpha, beta float32,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	eventLiteral := fmt.Sprintf("%s[%d]{0}", elementType, eventCount)
	queryLiteral := fmt.Sprintf("%s[%d]{0}", elementType, queryCount)
	outputLiteral := fmt.Sprintf("%s[%d]{0}", elementType, queryCount)
	entryLayout := fmt.Sprintf("%s,%s->%s", eventLiteral, queryLiteral, outputLiteral)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

%%add {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT result = %s[] add(lhs, rhs)
}

ENTRY main {
  events = %s parameter(0)
  queries = %s parameter(1)
  events_row = %s[%d,%d]{1,0} broadcast(events), dimensions={0,1}
  queries_col = %s[%d,%d]{1,0} broadcast(queries), dimensions={1,0}
  delta = %s subtract(queries_col, events_row)
  neg_beta = %s[] constant(%g)
  neg_beta_b = %s broadcast(neg_beta), dimensions={0,1}
  scaled = %s multiply(delta, neg_beta_b)
  exp_term = %s exponential(scaled)
  alpha_c = %s[] constant(%g)
  alpha_b = %s broadcast(alpha_c), dimensions={0,1}
  contrib = %s multiply(exp_term, alpha_b)
  valid = pred[%d,%d]{1,0} compare(events_row, queries_col), direction=LE
  zero = %s[] constant(0)
  zero_b = %s broadcast(zero), dimensions={0,1}
  masked = %s select(valid, contrib, zero_b)
  zero_s = %s[] constant(0)
  excitation = %s[%d]{0} reduce(masked, zero_s), dimensions={1}, to_apply=%%add
  mu_c = %s[] constant(%g)
  mu_b = %s broadcast(mu_c), dimensions={0}
  ROOT result = %s add(mu_b, excitation)
}
`, moduleName, entryLayout,
		elementType, elementType, elementType,
		eventLiteral, queryLiteral,
		eventLiteral, queryCount, eventCount,
		queryLiteral, queryCount, eventCount,
		outputLiteral, -beta, outputLiteral, outputLiteral, outputLiteral,
		alpha, outputLiteral, outputLiteral,
		queryCount, eventCount, elementType, outputLiteral, outputLiteral,
		elementType, queryLiteral, mu, queryLiteral, queryLiteral), nil
}

func RenderHawkesKernelMatrix(
	moduleName string,
	elementFormat dtype.DType,
	eventCount int,
	alpha, beta float32,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	eventLiteral := fmt.Sprintf("%s[%d]{0}", elementType, eventCount)
	outputLiteral := fmt.Sprintf("%s[%d,%d]{1,0}", elementType, eventCount, eventCount)
	entryLayout := fmt.Sprintf("%s->%s", eventLiteral, outputLiteral)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
  events = %s parameter(0)
  row = %s[%d,%d]{1,0} broadcast(events), dimensions={0,1}
  col = %s[%d,%d]{1,0} broadcast(events), dimensions={1,0}
  delta = %s subtract(row, col)
  neg_beta = %s[] constant(%g)
  neg_beta_b = %s broadcast(neg_beta), dimensions={0,1}
  scaled = %s multiply(delta, neg_beta_b)
  exp_term = %s exponential(scaled)
  alpha_c = %s[] constant(%g)
  alpha_b = %s broadcast(alpha_c), dimensions={0,1}
  contrib = %s multiply(exp_term, alpha_b)
  row_idx = s32[%d,%d]{1,0} reshape(iota(s32[%d]{0}), dimensions={%d,1})
  col_idx = s32[%d,%d]{1,0} reshape(iota(s32[%d]{0}), dimensions={1,%d})
  valid = pred[%d,%d]{1,0} compare(col_idx, row_idx), direction=LT
  zero = %s[] constant(0)
  zero_b = %s broadcast(zero), dimensions={0,1}
  ROOT result = %s select(valid, contrib, zero_b)
}
`, moduleName, entryLayout,
		eventLiteral,
		eventLiteral, eventCount, eventCount,
		eventLiteral, eventCount, eventCount,
		outputLiteral, -beta, outputLiteral, outputLiteral, outputLiteral,
		alpha, outputLiteral, outputLiteral,
		eventCount, eventCount, eventCount, eventCount,
		eventCount, eventCount, eventCount, eventCount,
		eventCount, eventCount, elementType, outputLiteral, outputLiteral), nil
}

func RenderHawkesLogLikelihood(
	moduleName string,
	elementFormat dtype.DType,
	eventCount int,
	totalT, mu, alpha, beta float32,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	eventLiteral := fmt.Sprintf("%s[%d]{0}", elementType, eventCount)
	scalarLiteral := fmt.Sprintf("%s[]", elementType)
	entryLayout := fmt.Sprintf("%s->%s", eventLiteral, scalarLiteral)
	minIntensity := float32(1e-12)
	alphaOverBeta := alpha / beta

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

%%add {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT result = %s[] add(lhs, rhs)
}

%%max {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT result = %s[] maximum(lhs, rhs)
}

ENTRY main {
  events = %s parameter(0)
  row = %s[%d,%d]{1,0} broadcast(events), dimensions={0,1}
  col = %s[%d,%d]{1,0} broadcast(events), dimensions={1,0}
  delta = %s subtract(row, col)
  neg_beta = %s[] constant(%g)
  neg_beta_b = %s broadcast(neg_beta), dimensions={0,1}
  scaled = %s multiply(delta, neg_beta_b)
  exp_term = %s exponential(scaled)
  alpha_c = %s[] constant(%g)
  alpha_b = %s broadcast(alpha_c), dimensions={0,1}
  contrib = %s multiply(exp_term, alpha_b)
  row_idx = s32[%d,%d]{1,0} reshape(iota(s32[%d]{0}), dimensions={%d,1})
  col_idx = s32[%d,%d]{1,0} reshape(iota(s32[%d]{0}), dimensions={1,%d})
  prior = pred[%d,%d]{1,0} compare(col_idx, row_idx), direction=LT
  zero = %s[] constant(0)
  zero_b = %s broadcast(zero), dimensions={0,1}
  masked = %s select(prior, contrib, zero_b)
  zero_s = %s[] constant(0)
  excitation_rows = %s[%d]{0} reduce(masked, zero_s), dimensions={1}, to_apply=%%add
  mu_c = %s[] constant(%g)
  mu_b = %s broadcast(mu_c), dimensions={0}
  intensity = %s add(mu_b, excitation_rows)
  min_c = %s[] constant(%g)
  min_b = %s broadcast(min_c), dimensions={0}
  clamped = %s maximum(intensity, min_b)
  log_intensity = %s log(clamped)
  zero_log = %s[] constant(0)
  log_sum = %s[] reduce(log_intensity, zero_log), dimensions={0}, to_apply=%%add
  total_t = %s[] constant(%g)
  total_b = %s broadcast(total_t), dimensions={0}
  tail = %s subtract(total_b, events)
  neg_beta_tail = %s multiply(tail, neg_beta)
  tail_exp = %s exponential(neg_beta_tail)
  one = %s[] constant(1)
  one_minus = %s subtract(one, tail_exp)
  alpha_beta = %s[] constant(%g)
  compensator_events = %s multiply(one_minus, alpha_beta)
  compensator_sum = %s reduce(compensator_events, zero_s), dimensions={0}, to_apply=%%add
  mu_total = %s multiply(mu_c, total_t)
  compensator = %s add(mu_total, compensator_sum)
  ROOT result = %s subtract(log_sum, compensator)
}
`, moduleName, entryLayout,
		elementType, elementType, elementType,
		elementType, elementType, elementType,
		eventLiteral,
		eventLiteral, eventCount, eventCount,
		eventLiteral, eventCount, eventCount,
		scalarLiteral, -beta, scalarLiteral, scalarLiteral, scalarLiteral,
		alpha, scalarLiteral, scalarLiteral,
		eventCount, eventCount, eventCount, eventCount,
		eventCount, eventCount, eventCount, eventCount,
		eventCount, eventCount, elementType, scalarLiteral, scalarLiteral,
		elementType, scalarLiteral, mu, scalarLiteral, scalarLiteral,
		elementType, minIntensity, scalarLiteral, scalarLiteral, scalarLiteral,
		elementType, scalarLiteral, scalarLiteral,
		totalT, scalarLiteral, scalarLiteral, scalarLiteral, scalarLiteral,
		elementType, alphaOverBeta, scalarLiteral, scalarLiteral, scalarLiteral, scalarLiteral), nil
}

func RenderMarkovMutualInformation(
	moduleName string,
	elementFormat dtype.DType,
	xCount, yCount int,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	jointLiteral := fmt.Sprintf("%s[%d,%d]{1,0}", elementType, xCount, yCount)
	scalarLiteral := fmt.Sprintf("%s[]", elementType)
	entryLayout := fmt.Sprintf("%s->%s", jointLiteral, scalarLiteral)
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
  joint = %s parameter(0)
  eps = %s[] constant(%g)
  denom = %s multiply(mx_b, my_b)
  eps_b = %s broadcast(eps), dimensions={0,1}
  denom_eps = %s maximum(denom, eps_b)
  ratio = %s divide(joint, denom_eps)
  log_ratio = %s log(ratio)
  weighted = %s multiply(joint, log_ratio)
  zero = %s[] constant(0)
  zero_b = %s broadcast(zero), dimensions={0,1}
  valid = pred[%d,%d]{1,0} compare(joint, eps_b), direction=GT
  masked = %s select(valid, weighted, zero_b)
  ROOT result = %s reduce(masked, zero), dimensions={0,1}, to_apply=%%add
}
`, moduleName, entryLayout,
		elementType, elementType, elementType,
		elementType, elementType, elementType,
		jointLiteral, elementType, epsilon,
		elementType, xCount, elementType,
		elementType, yCount, elementType,
		elementType, xCount, yCount,
		elementType, xCount, yCount,
		jointLiteral, jointLiteral, jointLiteral, jointLiteral,
		jointLiteral, jointLiteral, jointLiteral,
		elementType, xCount, yCount,
		jointLiteral, scalarLiteral), nil
}

func RenderMarkovBlanketPartition(
	moduleName string,
	nodeCount, internalCount int,
) (string, error) {
	adjacencyLiteral := fmt.Sprintf("f32[%d,%d]{1,0}", nodeCount, nodeCount)
	internalLiteral := fmt.Sprintf("s32[%d]{0}", internalCount)
	outputLiteral := fmt.Sprintf("s32[%d]{0}", nodeCount)
	entryLayout := fmt.Sprintf("%s,%s->%s", adjacencyLiteral, internalLiteral, outputLiteral)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

%%add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT result = f32[] add(lhs, rhs)
}

ENTRY main {
  adjacency = %s parameter(0)
  internal_list = %s parameter(1)
  zeros = f32[%d]{0} constant(0)
  ones = f32[%d]{0} broadcast(f32[] constant(1)), dimensions={0}
  idx_2d = s32[%d,1]{1,0} reshape(internal_list), new_dimensions={%d,1}
  mask_f32 = f32[%d]{0} scatter(zeros, idx_2d, ones),
    update_window_dims={},
    inserted_window_dims={0},
    scatter_dims_to_operand_dims={0},
    index_vector_dim=1,
    to_apply=%%add
  mask_pred = pred[%d]{0} compare(mask_f32, f32[] constant(0)), direction=GT
  row_mask = f32[%d,%d]{1,0} broadcast(mask_f32), dimensions={1}
  masked_in = f32[%d,%d]{1,0} multiply(adjacency, row_mask)
  incoming = f32[%d]{0} reduce(masked_in, f32[] constant(0)), dimensions={0}, to_apply=%%add
  col_mask = f32[%d,%d]{1,0} broadcast(mask_f32), dimensions={0}
  masked_out = f32[%d,%d]{1,0} multiply(adjacency, col_mask)
  outgoing = f32[%d]{0} reduce(masked_out, f32[] constant(0)), dimensions={1}, to_apply=%%add
  has_in = pred[%d]{0} compare(incoming, f32[] constant(0)), direction=NE
  has_out = pred[%d]{0} compare(outgoing, f32[] constant(0)), direction=NE
  both = pred[%d]{0} and(has_in, has_out)
  classified = s32[%d]{0} select(has_out, s32[%d]{0} broadcast(s32[] constant(1)), s32[%d]{0} broadcast(s32[] constant(3)))
  classified2 = s32[%d]{0} select(both, s32[%d]{0} broadcast(s32[] constant(2)), classified)
  ROOT result = s32[%d]{0} select(mask_pred, s32[%d]{0} broadcast(s32[] constant(0)), classified2)
}
`, moduleName, entryLayout,
		adjacencyLiteral, internalLiteral,
		nodeCount, internalCount, internalCount, internalCount, nodeCount,
		nodeCount, nodeCount, nodeCount, nodeCount, nodeCount, nodeCount,
		nodeCount, nodeCount, nodeCount, nodeCount, nodeCount, nodeCount,
		nodeCount, nodeCount, nodeCount, nodeCount, nodeCount, nodeCount,
		nodeCount, nodeCount, nodeCount, nodeCount, nodeCount, nodeCount), nil
}
