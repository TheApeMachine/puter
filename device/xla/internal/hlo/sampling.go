package hlo

import (
	"fmt"

	"github.com/theapemachine/manifesto/dtype"
)

func RenderGreedySample(
	moduleName string,
	elementFormat dtype.DType,
	vocabSize int,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	logitsLiteral := fmt.Sprintf("%s[%d]{0}", elementType, vocabSize)
	entryLayout := fmt.Sprintf("%s->s32[]", logitsLiteral)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

%%max {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT result = %s[] maximum(lhs, rhs)
}

%%min {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT result = s32[] minimum(lhs, rhs)
}

ENTRY main {
  logits = %s parameter(0)
  neg_inf = %s[] constant(-inf)
  row_max = %s[] reduce(logits, neg_inf), dimensions={0}, to_apply=%%max
  row_max_b = %s broadcast(row_max), dimensions={}
  eq = pred[%d]{0} compare(logits, row_max_b), direction=EQ
  indices = s32[%d]{0} convert(iota(s32[%d]{0}), dimensions={0})
  sentinel = s32[] constant(2147483647)
  sentinel_b = s32[%d]{0} broadcast(sentinel), dimensions={}
  masked = s32[%d]{0} select(eq, indices, sentinel_b)
  ROOT result = s32[] reduce(masked, s32[] constant(0)), dimensions={0}, to_apply=%%min
}
`, moduleName, entryLayout,
		elementType, elementType, elementType,
		logitsLiteral, elementType, elementType, logitsLiteral,
		vocabSize, vocabSize, vocabSize, vocabSize, vocabSize), nil
}

func RenderSoftmaxSort(
	moduleName string,
	elementFormat dtype.DType,
	vocabSize int,
	temperature float32,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	if temperature == 0 {
		temperature = 1
	}

	logitsLiteral := fmt.Sprintf("%s[%d]{0}", elementType, vocabSize)
	stackLiteral := fmt.Sprintf("%s[%d]{0}", elementType, vocabSize*2)
	entryLayout := fmt.Sprintf("%s->%s", logitsLiteral, stackLiteral)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

%%max {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT result = %s[] maximum(lhs, rhs)
}

%%add {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT result = %s[] add(lhs, rhs)
}

%%compare {
  lhs_prob = %s[] get-tuple-element((%s[], s32[]) parameter(0), index=0)
  rhs_prob = %s[] get-tuple-element((%s[], s32[]) parameter(1), index=0)
  ROOT result = pred[] compare(lhs_prob, rhs_prob), direction=GT
}

ENTRY main {
  logits = %s parameter(0)
  neg_inf = %s[] constant(-inf)
  row_max = %s[] reduce(logits, neg_inf), dimensions={0}, to_apply=%%max
  row_max_b = %s broadcast(row_max), dimensions={}
  centered = %s subtract(logits, row_max_b)
  temp = %s[] constant(%g)
  temp_b = %s broadcast(temp), dimensions={}
  scaled = %s divide(centered, temp_b)
  exp_vals = %s exponential(scaled)
  zero = %s[] constant(0)
  denom = %s[] reduce(exp_vals, zero), dimensions={0}, to_apply=%%add
  denom_b = %s broadcast(denom), dimensions={}
  probs = %s divide(exp_vals, denom_b)
  indices = s32[%d]{0} iota(), iota_dimension=0
  sorted = (%s[%d], s32[%d]) sort(probs, indices), dimensions={0}, is_stable=true, to_apply=%%compare
  sorted_probs = %s[%d]{0} get-tuple-element(sorted, index=0)
  sorted_indices = s32[%d]{0} get-tuple-element(sorted, index=1)
  sorted_indices_f32 = %s convert(sorted_indices)
  ROOT result = %s concatenate(sorted_probs, sorted_indices_f32), dimensions={0}
}
`, moduleName, entryLayout,
		elementType, elementType, elementType,
		elementType, elementType, elementType,
		elementType, elementType, elementType, elementType,
		logitsLiteral, elementType, elementType, logitsLiteral,
		logitsLiteral, elementType, temperature, logitsLiteral,
		logitsLiteral, logitsLiteral, elementType, elementType,
		logitsLiteral, logitsLiteral, vocabSize,
		elementType, vocabSize, vocabSize,
		elementType, vocabSize,
		vocabSize,
		logitsLiteral, stackLiteral), nil
}

func RenderProbabilisticSample(
	moduleName string,
	elementFormat dtype.DType,
	vocabSize int,
	operationName string,
	temperature float32,
	target float32,
	topK int,
	topP float32,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	if temperature == 0 {
		temperature = 1
	}

	if topK <= 0 || topK > vocabSize {
		topK = vocabSize
	}

	if topP <= 0 {
		topP = 1
	}

	if topP > 1 {
		topP = 1
	}

	switch operationName {
	case "topk_sample":
		return renderTopKSample(moduleName, elementType, vocabSize, temperature, target, topK), nil
	case "topp_sample":
		return renderTopPSample(moduleName, elementType, vocabSize, temperature, target, topP), nil
	default:
		return "", fmt.Errorf("unsupported XLA probabilistic sample: %s", operationName)
	}
}

func renderSamplingPrefix(moduleName string, elementType string, vocabSize int, temperature float32) string {
	logitsLiteral := fmt.Sprintf("%s[%d]{0}", elementType, vocabSize)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s->s32[]}

%%max {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT result = f32[] maximum(lhs, rhs)
}

%%add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT result = f32[] add(lhs, rhs)
}

%%min_pair {
  lhs_pos = s32[] parameter(0)
  lhs_idx = s32[] parameter(1)
  rhs_pos = s32[] parameter(2)
  rhs_idx = s32[] parameter(3)
  take_lhs = pred[] compare(lhs_pos, rhs_pos), direction=LE
  pos = s32[] select(take_lhs, lhs_pos, rhs_pos)
  idx = s32[] select(take_lhs, lhs_idx, rhs_idx)
  ROOT result = (s32[], s32[]) tuple(pos, idx)
}

%%compare {
  lhs_prob = f32[] get-tuple-element((f32[], s32[]) parameter(0), index=0)
  rhs_prob = f32[] get-tuple-element((f32[], s32[]) parameter(1), index=0)
  ROOT result = pred[] compare(lhs_prob, rhs_prob), direction=GT
}

ENTRY main {
  logits = %s parameter(0)
  logits_f32 = f32[%d]{0} convert(logits)
  neg_inf = f32[] constant(-inf)
  row_max = f32[] reduce(logits_f32, neg_inf), dimensions={0}, to_apply=%%max
  row_max_b = f32[%d]{0} broadcast(row_max), dimensions={}
  centered = f32[%d]{0} subtract(logits_f32, row_max_b)
  temp = f32[] constant(%g)
  temp_b = f32[%d]{0} broadcast(temp), dimensions={}
  scaled = f32[%d]{0} divide(centered, temp_b)
  exp_vals = f32[%d]{0} exponential(scaled)
  zero = f32[] constant(0)
  denom = f32[] reduce(exp_vals, zero), dimensions={0}, to_apply=%%add
  denom_b = f32[%d]{0} broadcast(denom), dimensions={}
  probs = f32[%d]{0} divide(exp_vals, denom_b)
  indices = s32[%d]{0} iota(), iota_dimension=0
  sorted = (f32[%d], s32[%d]) sort(probs, indices), dimensions={0}, is_stable=true, to_apply=%%compare
  sorted_probs = f32[%d]{0} get-tuple-element(sorted, index=0)
  sorted_indices = s32[%d]{0} get-tuple-element(sorted, index=1)
  cumulative = f32[%d]{0} reduce-window(sorted_probs, zero), window={size=%d pad=%d_0}, to_apply=%%add
`, moduleName, logitsLiteral,
		logitsLiteral, vocabSize, vocabSize, vocabSize,
		temperature, vocabSize, vocabSize, vocabSize, vocabSize,
		vocabSize, vocabSize, vocabSize, vocabSize, vocabSize,
		vocabSize, vocabSize, vocabSize, vocabSize, vocabSize-1)
}

func renderTopKSample(
	moduleName string,
	elementType string,
	vocabSize int,
	temperature float32,
	target float32,
	topK int,
) string {
	prefix := renderSamplingPrefix(moduleName, elementType, vocabSize, temperature)

	return fmt.Sprintf(`%s  positions = s32[%d]{0} iota(), iota_dimension=0
  topk_c = s32[] constant(%d)
  topk_b = s32[%d]{0} broadcast(topk_c), dimensions={}
  in_topk = pred[%d]{0} compare(positions, topk_b), direction=LT
  zero_b = f32[%d]{0} broadcast(zero), dimensions={}
  masked_probs = f32[%d]{0} select(in_topk, sorted_probs, zero_b)
  mass = f32[] reduce(masked_probs, zero), dimensions={0}, to_apply=%%add
  target = f32[] constant(%g)
  threshold = f32[] multiply(target, mass)
  threshold_b = f32[%d]{0} broadcast(threshold), dimensions={}
  masked_cumulative = f32[%d]{0} reduce-window(masked_probs, zero), window={size=%d pad=%d_0}, to_apply=%%add
  crosses = pred[%d]{0} compare(masked_cumulative, threshold_b), direction=GE
  candidate = pred[%d]{0} and(in_topk, crosses)
  sentinel = s32[] constant(2147483647)
  sentinel_b = s32[%d]{0} broadcast(sentinel), dimensions={}
  candidate_pos = s32[%d]{0} select(candidate, positions, sentinel_b)
  candidate_idx = s32[%d]{0} select(candidate, sorted_indices, sentinel_b)
  selected = (s32[], s32[]) reduce(candidate_pos, candidate_idx, sentinel, sentinel), dimensions={0}, to_apply=%%min_pair
  ROOT result = s32[] get-tuple-element(selected), index=1
}
`, prefix, vocabSize, topK, vocabSize, vocabSize, vocabSize, vocabSize,
		target, vocabSize, vocabSize, vocabSize, vocabSize-1, vocabSize,
		vocabSize, vocabSize, vocabSize, vocabSize)
}

func renderTopPSample(
	moduleName string,
	elementType string,
	vocabSize int,
	temperature float32,
	target float32,
	topP float32,
) string {
	prefix := renderSamplingPrefix(moduleName, elementType, vocabSize, temperature)

	return fmt.Sprintf(`%s  positions = s32[%d]{0} iota(), iota_dimension=0
  top_p = f32[] constant(%g)
  top_p_b = f32[%d]{0} broadcast(top_p), dimensions={}
  previous = f32[%d]{0} subtract(cumulative, sorted_probs)
  prefix_mask = pred[%d]{0} compare(previous, top_p_b), direction=LT
  zero_b = f32[%d]{0} broadcast(zero), dimensions={}
  masked_probs = f32[%d]{0} select(prefix_mask, sorted_probs, zero_b)
  mass = f32[] reduce(masked_probs, zero), dimensions={0}, to_apply=%%add
  target = f32[] constant(%g)
  threshold = f32[] multiply(target, mass)
  threshold_b = f32[%d]{0} broadcast(threshold), dimensions={}
  masked_cumulative = f32[%d]{0} reduce-window(masked_probs, zero), window={size=%d pad=%d_0}, to_apply=%%add
  crosses = pred[%d]{0} compare(masked_cumulative, threshold_b), direction=GE
  candidate = pred[%d]{0} and(prefix_mask, crosses)
  sentinel = s32[] constant(2147483647)
  sentinel_b = s32[%d]{0} broadcast(sentinel), dimensions={}
  candidate_pos = s32[%d]{0} select(candidate, positions, sentinel_b)
  candidate_idx = s32[%d]{0} select(candidate, sorted_indices, sentinel_b)
  selected = (s32[], s32[]) reduce(candidate_pos, candidate_idx, sentinel, sentinel), dimensions={0}, to_apply=%%min_pair
  ROOT result = s32[] get-tuple-element(selected), index=1
}
`, prefix, vocabSize, topP, vocabSize, vocabSize, vocabSize, vocabSize,
		vocabSize, target, vocabSize, vocabSize, vocabSize, vocabSize-1,
		vocabSize, vocabSize, vocabSize, vocabSize, vocabSize)
}
