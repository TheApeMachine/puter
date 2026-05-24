package hlo

import (
	"fmt"

	"github.com/theapemachine/manifesto/dtype"
)

func RenderRoPEPairs(
	moduleName string,
	elementFormat dtype.DType,
	headDim int,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	halfDim := headDim / 2
	inputLiteral := fmt.Sprintf("%s[%d]{0}", elementType, headDim)
	halfLiteral := fmt.Sprintf("%s[%d]{0}", elementType, halfDim)
	pairLiteral := fmt.Sprintf("%s[%d,2]{1,0}", elementType, halfDim)
	colLiteral := fmt.Sprintf("%s[%d,1]{1,0}", elementType, halfDim)
	entryLayout := fmt.Sprintf("%s,%s,%s->%s", inputLiteral, halfLiteral, halfLiteral, inputLiteral)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
  input = %s parameter(0)
  cos_buf = %s parameter(1)
  sin_buf = %s parameter(2)
  pairs = %s reshape(input), dimensions={%d,2}
  even = %s reshape(%s slice(pairs, {0,0}, {%d,1}, {1,1})), dimensions={%d}
  odd = %s reshape(%s slice(pairs, {0,1}, {%d,2}, {1,1})), dimensions={%d}
  out_even = %s subtract(%s multiply(even, cos_buf), %s multiply(odd, sin_buf))
  out_odd = %s add(%s multiply(even, sin_buf), %s multiply(odd, cos_buf))
  even_col = %s reshape(out_even), dimensions={%d,1}
  odd_col = %s reshape(out_odd), dimensions={%d,1}
  merged = %s concatenate({even_col, odd_col}), dimensions={1}
  ROOT result = %s reshape(merged), dimensions={%d}
}
`, moduleName, entryLayout,
		inputLiteral, halfLiteral, halfLiteral,
		pairLiteral, halfDim,
		halfLiteral, pairLiteral, halfDim, halfDim,
		halfLiteral, pairLiteral, halfDim, halfDim,
		halfLiteral, halfLiteral, halfLiteral,
		halfLiteral, halfLiteral, halfLiteral,
		colLiteral, halfDim,
		colLiteral, halfDim,
		pairLiteral,
		inputLiteral, headDim), nil
}

func RenderRoPE(
	moduleName string,
	elementFormat dtype.DType,
	seqLen, numHeads, headDim int,
	baseFreq float64,
	startPosition int,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	halfDim := headDim / 2
	inputLiteral := fmt.Sprintf("%s[%d,%d,%d]{2,1,0}", elementType, seqLen, numHeads, headDim)
	seqLiteral := fmt.Sprintf("%s[%d]{0}", elementType, seqLen)
	halfLiteral := fmt.Sprintf("%s[%d]{0}", elementType, halfDim)
	tableLiteral := fmt.Sprintf("%s[%d,%d]{1,0}", elementType, seqLen, halfDim)
	broadcastLiteral := fmt.Sprintf("%s[%d,%d,%d]{2,1,0}", elementType, seqLen, numHeads, halfDim)
	pairLiteral := fmt.Sprintf("%s[%d,%d,%d,2]{3,2,1,0}", elementType, seqLen, numHeads, halfDim)
	colLiteral := fmt.Sprintf("%s[%d,%d,%d,1]{3,2,1,0}", elementType, seqLen, numHeads, halfDim)
	entryLayout := fmt.Sprintf("%s->%s", inputLiteral, inputLiteral)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
  input = %s parameter(0)
  pos_iota = iota(s32[%d]{0}), dimensions={0}
  start = s32[] constant(%d)
  start_b = s32[%d]{0} broadcast(start), dimensions={}
  pos_i32 = s32[%d]{0} add(pos_iota, start_b)
  pos = %s convert(pos_i32)
  pair_iota = iota(s32[%d]{0}), dimensions={0}
  pair_idx = %s convert(pair_iota)
  two = %s[] constant(2)
  head_dim = %s[] constant(%d)
  exp_num = %s multiply(two, pair_idx)
  exp_den = %s convert(head_dim)
  exponent = %s negate(%s divide(exp_num, exp_den))
  base = %s[] constant(%g)
  inv_freq = %s power(base, exponent)
  pos_col = %s reshape(pos), dimensions={%d,1}
  inv_row = %s reshape(inv_freq), dimensions={1,%d}
  theta = %s multiply(pos_col, inv_row)
  cos_table = %s cosine(theta)
  sin_table = %s sine(theta)
  cos_b = %s broadcast(cos_table), dimensions={0,2}
  sin_b = %s broadcast(sin_table), dimensions={0,2}
  pairs = %s reshape(input), dimensions={%d,%d,%d,2}
  even = %s reshape(%s slice(pairs, {0,0,0,0}, {%d,%d,%d,1}, {1,1,1,1})), dimensions={%d,%d,%d}
  odd = %s reshape(%s slice(pairs, {0,0,0,1}, {%d,%d,%d,2}, {1,1,1,1})), dimensions={%d,%d,%d}
  out_even = %s subtract(%s multiply(even, cos_b), %s multiply(odd, sin_b))
  out_odd = %s add(%s multiply(even, sin_b), %s multiply(odd, cos_b))
  even_col = %s reshape(out_even), dimensions={%d,%d,%d,1}
  odd_col = %s reshape(out_odd), dimensions={%d,%d,%d,1}
  merged = %s concatenate({even_col, odd_col}), dimensions={3}
  ROOT result = %s reshape(merged), dimensions={%d,%d,%d}
}
`, moduleName, entryLayout,
		inputLiteral,
		seqLen, startPosition, seqLen, seqLen,
		elementType,
		halfDim, elementType,
		elementType,
		elementType, headDim,
		halfLiteral,
		elementType,
		halfLiteral, halfLiteral,
		elementType, baseFreq, halfLiteral,
		seqLiteral, seqLen, halfLiteral, halfDim, tableLiteral,
		tableLiteral, tableLiteral,
		broadcastLiteral, broadcastLiteral,
		pairLiteral, seqLen, numHeads, halfDim,
		broadcastLiteral, pairLiteral, seqLen, numHeads, halfDim, seqLen, numHeads, halfDim,
		broadcastLiteral, pairLiteral, seqLen, numHeads, halfDim, seqLen, numHeads, halfDim,
		broadcastLiteral, broadcastLiteral, broadcastLiteral,
		broadcastLiteral, broadcastLiteral, broadcastLiteral,
		colLiteral, seqLen, numHeads, halfDim,
		colLiteral, seqLen, numHeads, halfDim,
		pairLiteral,
		inputLiteral, seqLen, numHeads, headDim), nil
}
