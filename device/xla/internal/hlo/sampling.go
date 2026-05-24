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
