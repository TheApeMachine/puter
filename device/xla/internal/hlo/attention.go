package hlo

import (
	"fmt"
	"math"

	"github.com/theapemachine/manifesto/dtype"
)

func RenderScaledDotProductAttention(
	moduleName string,
	elementFormat dtype.DType,
	seqQ, seqK, depth, valueDim int,
	causal bool,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	queryLiteral := fmt.Sprintf("%s[%d,%d]{1,0}", elementType, seqQ, depth)
	keyLiteral := fmt.Sprintf("%s[%d,%d]{1,0}", elementType, seqK, depth)
	valueLiteral := fmt.Sprintf("%s[%d,%d]{1,0}", elementType, seqK, valueDim)
	outputLiteral := fmt.Sprintf("%s[%d,%d]{1,0}", elementType, seqQ, valueDim)
	scoreLiteral := fmt.Sprintf("%s[%d,%d]{1,0}", elementType, seqQ, seqK)
	rowLiteral := fmt.Sprintf("%s[%d]{0}", elementType, seqQ)
	scale := 1.0 / math.Sqrt(float64(depth))
	entryLayout := fmt.Sprintf("%s,%s,%s->%s", queryLiteral, keyLiteral, valueLiteral, outputLiteral)

	causalBlock := ""
	scoreSource := "scaled"

	if causal {
		causalBlock = fmt.Sprintf(`  q_idx = %s reshape(iota(s32[%d]{0}), dimensions={%d,1})
  k_idx = s32[%d,%d]{1,0} reshape(iota(s32[%d]{0}), dimensions={1,%d})
  causal_mask = pred[%d,%d]{1,0} compare(k_idx, q_idx), direction=GT
  neg_inf = %s[] constant(-inf)
  neg_inf_b = %s broadcast(neg_inf), dimensions={0,1}
  scaled_masked = %s select(causal_mask, neg_inf_b, scaled)
`, scoreLiteral, seqQ, seqQ, seqK, seqK, seqK, seqK, seqQ, seqK,
			elementType, scoreLiteral, scoreLiteral)
		scoreSource = "scaled_masked"
	}

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

ENTRY main {
  query = %s parameter(0)
  key = %s parameter(1)
  value = %s parameter(2)
  key_t = %s transpose(key), dimensions={1,0}
  scores = %s dot(query, key_t), lhs_contracting_dimensions={1}, rhs_contracting_dimensions={0}
  scale_c = %s[] constant(%g)
  scaled = %s multiply(scores, scale_c)
%s  neg_inf = %s[] constant(-inf)
  row_max = %s reduce(%s, neg_inf), dimensions={1}, to_apply=%%max
  row_max_b = %s broadcast(row_max), dimensions={0}
  shifted = %s subtract(%s, row_max_b)
  exp_val = %s exponential(shifted)
  zero = %s[] constant(0)
  row_sum = %s reduce(exp_val, zero), dimensions={1}, to_apply=%%add
  row_sum_b = %s broadcast(row_sum), dimensions={0}
  weights = %s divide(exp_val, row_sum_b)
  ROOT result = %s dot(weights, value), lhs_contracting_dimensions={1}, rhs_contracting_dimensions={0}
}
`, moduleName, entryLayout,
		elementType, elementType, elementType,
		elementType, elementType, elementType,
		queryLiteral, keyLiteral, valueLiteral,
		keyLiteral, scoreLiteral,
		elementType, scale, scoreLiteral,
		causalBlock,
		elementType, rowLiteral, scoreSource, scoreLiteral, scoreLiteral, scoreSource, scoreLiteral,
		elementType, rowLiteral, scoreLiteral, scoreLiteral,
		scoreLiteral, outputLiteral), nil
}

func RenderMultiHeadAttention(
	moduleName string,
	elementFormat dtype.DType,
	seqQ, seqK, numHeads, kvHeads, headDim int,
	causal bool,
	windowSize int,
	alibiSlope float32,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	if kvHeads <= 0 {
		kvHeads = numHeads
	}

	if numHeads%kvHeads != 0 {
		return "", fmt.Errorf("multi-head attention head count must divide kv head count")
	}

	queryLiteral := fmt.Sprintf("%s[%d,%d]{1,0}", elementType, seqQ, numHeads*headDim)
	keyLiteral := fmt.Sprintf("%s[%d,%d]{1,0}", elementType, seqK, kvHeads*headDim)
	valueLiteral := fmt.Sprintf("%s[%d,%d]{1,0}", elementType, seqK, kvHeads*headDim)
	outputLiteral := fmt.Sprintf("%s[%d,%d]{1,0}", elementType, seqQ, numHeads*headDim)
	batchQueryLiteral := fmt.Sprintf("%s[%d,%d,%d]{2,1,0}", elementType, numHeads, seqQ, headDim)
	batchKeyLiteral := fmt.Sprintf("%s[%d,%d,%d]{2,1,0}", elementType, kvHeads, seqK, headDim)
	batchValueLiteral := fmt.Sprintf("%s[%d,%d,%d]{2,1,0}", elementType, kvHeads, seqK, headDim)
	batchOutputLiteral := fmt.Sprintf("%s[%d,%d,%d]{2,1,0}", elementType, numHeads, seqQ, headDim)
	scoreLiteral := fmt.Sprintf("%s[%d,%d,%d]{2,1,0}", elementType, numHeads, seqQ, seqK)
	rowLiteral := fmt.Sprintf("%s[%d,%d]{1,0}", elementType, numHeads, seqQ)
	scale := 1.0 / math.Sqrt(float64(headDim))
	entryLayout := fmt.Sprintf("%s,%s,%s->%s", queryLiteral, keyLiteral, valueLiteral, outputLiteral)

	repeatFactor := numHeads / kvHeads
	expandKey := batchKeyLiteral

	if repeatFactor > 1 {
		expandKey = fmt.Sprintf("%s[%d,%d,%d,%d]{3,2,1,0}", elementType, kvHeads, repeatFactor, seqK, headDim)
	}

	causalBlock := ""
	windowBlock := ""
	alibiBlock := ""
	scoreSource := "scaled"

	if causal {
		causalBlock = fmt.Sprintf(`  q_idx = %s reshape(iota(s32[%d]{0}), dimensions={1,%d,1})
  k_idx = s32[%d,%d,%d]{2,1,0} reshape(iota(s32[%d]{0}), dimensions={1,1,%d})
  causal_mask = pred[%d,%d,%d]{2,1,0} compare(k_idx, q_idx), direction=GT
  neg_inf = %s[] constant(-inf)
  neg_inf_b = %s broadcast(neg_inf), dimensions={0,1,2}
  scaled = %s select(causal_mask, neg_inf_b, scaled)
`, scoreLiteral, seqQ, seqQ, numHeads, seqQ, seqK, seqK, seqK, numHeads, seqQ, seqK,
			elementType, scoreLiteral, scoreLiteral)
	}

	if windowSize > 0 {
		windowBlock = fmt.Sprintf(`  q_idx_w = %s reshape(iota(s32[%d]{0}), dimensions={1,%d,1})
  k_idx_w = s32[%d,%d,%d]{2,1,0} reshape(iota(s32[%d]{0}), dimensions={1,1,%d})
  distance = s32[%d,%d,%d]{2,1,0} subtract(q_idx_w, k_idx_w)
  window = s32[] constant(%d)
  window_b = s32[%d,%d,%d]{2,1,0} broadcast(window), dimensions={0,1,2}
  window_mask = pred[%d,%d,%d]{2,1,0} compare(distance, window_b), direction=GE
  neg_inf_w = %s[] constant(-inf)
  neg_inf_w_b = %s broadcast(neg_inf_w), dimensions={0,1,2}
  scaled = %s select(window_mask, neg_inf_w_b, scaled)
`, scoreLiteral, seqQ, seqQ, numHeads, seqQ, seqK, seqK, seqK, numHeads, seqQ, seqK,
			numHeads, seqQ, seqK, windowSize, numHeads, seqQ, seqK, numHeads, seqQ, seqK,
			elementType, scoreLiteral, scoreLiteral)
	}

	if alibiSlope != 0 {
		alibiBlock = fmt.Sprintf(`  q_idx_a = %s convert(%s reshape(iota(s32[%d]{0}), dimensions={1,%d,1})), newtype=%s
  k_idx_a = %s convert(s32[%d,%d,%d]{2,1,0} reshape(iota(s32[%d]{0}), dimensions={1,1,%d})), newtype=%s
  alibi = %s multiply(%s[] constant(%g), %s subtract(k_idx_a, q_idx_a))
  scaled = %s add(scaled, alibi)
`, elementType, scoreLiteral, seqQ, seqQ, elementType,
			elementType, numHeads, seqQ, seqK, seqK, seqK, elementType,
			scoreLiteral, elementType, alibiSlope, scoreLiteral,
			scoreLiteral)
	}

	_ = scoreSource

	keyPrep := fmt.Sprintf(`  key_batched = %s reshape(key), dimensions={%d,%d,%d}
  value_batched = %s reshape(value), dimensions={%d,%d,%d}`, batchKeyLiteral, seqK, kvHeads, headDim, batchValueLiteral, seqK, kvHeads, headDim)

	if repeatFactor > 1 {
		keyPrep = fmt.Sprintf(`  key_batched = %s reshape(key), dimensions={%d,%d,%d}
  value_batched = %s reshape(value), dimensions={%d,%d,%d}
  key_expanded = %s reshape(key_batched), dimensions={%d,%d,1,%d,%d}
  value_expanded = %s reshape(value_batched), dimensions={%d,%d,1,%d,%d}
  key_tiled = %s broadcast(key_expanded), dimensions={0,1,2,3,4}
  value_tiled = %s broadcast(value_expanded), dimensions={0,1,2,3,4}
  key_batched = %s reshape(key_tiled), dimensions={%d,%d,%d}
  value_batched = %s reshape(value_tiled), dimensions={%d,%d,%d}`,
			batchKeyLiteral, seqK, kvHeads, headDim, batchValueLiteral, seqK, kvHeads, headDim,
			expandKey, kvHeads, seqK, repeatFactor, headDim,
			expandKey, kvHeads, seqK, repeatFactor, headDim,
			expandKey, expandKey,
			batchKeyLiteral, numHeads, seqK, headDim,
			batchValueLiteral, numHeads, seqK, headDim)
	}

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

ENTRY main {
  query = %s parameter(0)
  key = %s parameter(1)
  value = %s parameter(2)
  query_batched = %s reshape(query), dimensions={%d,%d,%d}
  %s
  query_heads = %s transpose(query_batched), dimensions={1,0,2}
  key_heads = %s transpose(key_batched), dimensions={1,0,2}
  value_heads = %s transpose(value_batched), dimensions={1,0,2}
  key_t = %s transpose(key_heads), dimensions={0,2,1}
  scores = %s dot(query_heads, key_t), lhs_batch_dimensions={0}, lhs_contracting_dimensions={2}, rhs_batch_dimensions={0}, rhs_contracting_dimensions={2}
  scale_c = %s[] constant(%g)
  scaled = %s multiply(scores, scale_c)
%s%s%s  neg_inf = %s[] constant(-inf)
  row_max = %s reduce(scaled, neg_inf), dimensions={2}, to_apply=%%max
  row_max_b = %s broadcast(row_max), dimensions={0,1}
  shifted = %s subtract(scaled, row_max_b)
  exp_val = %s exponential(shifted)
  zero = %s[] constant(0)
  row_sum = %s reduce(exp_val, zero), dimensions={2}, to_apply=%%add
  row_sum_b = %s broadcast(row_sum), dimensions={0,1}
  weights = %s divide(exp_val, row_sum_b)
  attended = %s dot(weights, value_heads), lhs_batch_dimensions={0}, lhs_contracting_dimensions={2}, rhs_batch_dimensions={0}, rhs_contracting_dimensions={1}
  attended_t = %s transpose(attended), dimensions={1,0,2}
  ROOT result = %s reshape(attended_t), dimensions={%d,%d}
}
`, moduleName, entryLayout,
		elementType, elementType, elementType,
		elementType, elementType, elementType,
		queryLiteral, keyLiteral, valueLiteral,
		batchQueryLiteral, seqQ, numHeads, headDim,
		keyPrep,
		batchQueryLiteral, batchKeyLiteral, batchValueLiteral,
		batchKeyLiteral, scoreLiteral,
		elementType, scale, scoreLiteral,
		causalBlock, windowBlock, alibiBlock,
		elementType, rowLiteral, scoreLiteral, scoreLiteral, scoreLiteral,
		elementType, rowLiteral, scoreLiteral, scoreLiteral,
		scoreLiteral, batchOutputLiteral, batchOutputLiteral,
		outputLiteral, seqQ, numHeads*headDim), nil
}
