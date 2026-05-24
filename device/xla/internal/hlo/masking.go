package hlo

import (
	"fmt"

	"github.com/theapemachine/manifesto/dtype"
)

func RenderCausalMask(
	moduleName string,
	elementFormat dtype.DType,
	seqQ, seqK int,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	outputLiteral := fmt.Sprintf("%s[%d,%d]{1,0}", elementType, seqQ, seqK)
	entryLayout := fmt.Sprintf("->%s", outputLiteral)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
  q_idx = s32[%d,%d]{1,0} reshape(iota(s32[%d]{0}), dimensions={%d,1})
  k_idx = s32[%d,%d]{1,0} reshape(iota(s32[%d]{0}), dimensions={1,%d})
  causal_mask = pred[%d,%d]{1,0} compare(k_idx, q_idx), direction=GT
  neg_inf = %s[] constant(-inf)
  neg_inf_b = %s broadcast(neg_inf), dimensions={0,1}
  zero = %s[] constant(0)
  zero_b = %s broadcast(zero), dimensions={0,1}
  ROOT result = %s select(causal_mask, neg_inf_b, zero_b)
}
`, moduleName, entryLayout,
		seqQ, seqQ, seqQ, seqQ, seqK, seqK, seqK, seqK, seqQ, seqK,
		elementType, outputLiteral, elementType, outputLiteral, outputLiteral), nil
}

func RenderALiBiBias(
	moduleName string,
	elementFormat dtype.DType,
	seqQ, seqK int,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	scoreLiteral := fmt.Sprintf("%s[%d,%d]{1,0}", elementType, seqQ, seqK)
	slopeLiteral := fmt.Sprintf("%s[1]{0}", elementType)
	entryLayout := fmt.Sprintf("%s,%s->%s", scoreLiteral, slopeLiteral, scoreLiteral)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
  scores = %s parameter(0)
  slope = %s parameter(1)
  q_idx = %s convert(s32[%d,%d]{1,0} reshape(iota(s32[%d]{0}), dimensions={%d,1}))
  k_idx = %s convert(s32[%d,%d]{1,0} reshape(iota(s32[%d]{0}), dimensions={1,%d}))
  distance = %s subtract(q_idx, k_idx)
  zero = %s[] constant(0)
  zero_b = %s broadcast(zero), dimensions={0,1}
  past = pred[%d,%d]{1,0} compare(distance, zero_b), direction=LT
  slope_b = %s broadcast(slope), dimensions={0,1}
  bias = %s multiply(slope_b, distance)
  adjusted = %s subtract(scores, bias)
  ROOT result = %s select(past, scores, adjusted)
}
`, moduleName, entryLayout,
		scoreLiteral, slopeLiteral,
		elementType, seqQ, seqQ, seqQ, seqQ,
		elementType, seqQ, seqK, seqK, seqK,
		scoreLiteral,
		elementType, scoreLiteral, seqQ, seqK,
		scoreLiteral, scoreLiteral, scoreLiteral, scoreLiteral), nil
}
