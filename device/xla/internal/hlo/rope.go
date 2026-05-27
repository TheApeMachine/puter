package hlo

import (
	"fmt"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device"
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
	mode int,
	scaling int,
	scalingFactor float64,
	lowFreqFactor float64,
	highFreqFactor float64,
	originalContext int,
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
	entryLayout := fmt.Sprintf("%s->%s", inputLiteral, inputLiteral)
	frequencyBody, err := renderRoPEFrequencyBody(
		elementType,
		halfLiteral,
		halfDim,
		baseFreq,
		scaling,
		scalingFactor,
		lowFreqFactor,
		highFreqFactor,
		originalContext,
	)

	if err != nil {
		return "", err
	}

	rotationBody, err := renderRoPERotationBody(
		elementType,
		inputLiteral,
		broadcastLiteral,
		seqLen,
		numHeads,
		headDim,
		halfDim,
		mode,
	)

	if err != nil {
		return "", err
	}

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
%s
  pos_col = %s reshape(pos), dimensions={%d,1}
  inv_row = %s reshape(inv_freq), dimensions={1,%d}
  theta = %s multiply(pos_col, inv_row)
  cos_table = %s cosine(theta)
  sin_table = %s sine(theta)
  cos_b = %s broadcast(cos_table), dimensions={0,2}
  sin_b = %s broadcast(sin_table), dimensions={0,2}
%s
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
		frequencyBody,
		seqLiteral, seqLen, halfLiteral, halfDim, tableLiteral,
		tableLiteral, tableLiteral,
		broadcastLiteral, broadcastLiteral,
		rotationBody), nil
}

func renderRoPEFrequencyBody(
	elementType string,
	halfLiteral string,
	halfDim int,
	baseFreq float64,
	scaling int,
	scalingFactor float64,
	lowFreqFactor float64,
	highFreqFactor float64,
	originalContext int,
) (string, error) {
	if scaling == int(device.RoPEScalingNone) {
		return fmt.Sprintf(`  base = %s[] constant(%g)
  inv_freq = %s power(base, exponent)`, elementType, baseFreq, halfLiteral), nil
	}

	if scaling != int(device.RoPEScalingLlama3) {
		return "", fmt.Errorf("xla rope: unsupported scaling %d", scaling)
	}

	predicateLiteral := fmt.Sprintf("pred[%d]{0}", halfDim)

	return fmt.Sprintf(`  base = %s[] constant(%g)
  inv_freq_raw = %s power(base, exponent)
  two_pi = %s[] constant(6.283185307179586)
  two_pi_b = %s broadcast(two_pi), dimensions={}
  wavelength = %s divide(two_pi_b, inv_freq_raw)
  factor = %s[] constant(%g)
  factor_b = %s broadcast(factor), dimensions={}
  scaled_inv_freq = %s divide(inv_freq_raw, factor_b)
  original = %s[] constant(%g)
  original_b = %s broadcast(original), dimensions={}
  low_factor = %s[] constant(%g)
  low_factor_b = %s broadcast(low_factor), dimensions={}
  high_factor = %s[] constant(%g)
  high_factor_b = %s broadcast(high_factor), dimensions={}
  low_wavelength = %s[] constant(%g)
  low_wavelength_b = %s broadcast(low_wavelength), dimensions={}
  high_wavelength = %s[] constant(%g)
  high_wavelength_b = %s broadcast(high_wavelength), dimensions={}
  low_band = %s compare(wavelength, low_wavelength_b), direction=GT
  high_band = %s compare(wavelength, high_wavelength_b), direction=LT
  smooth_num_raw = %s divide(original_b, wavelength)
  smooth_num = %s subtract(smooth_num_raw, low_factor_b)
  smooth_den = %s subtract(high_factor_b, low_factor_b)
  smooth = %s divide(smooth_num, smooth_den)
  one = %s[] constant(1)
  one_b = %s broadcast(one), dimensions={}
  inverse_smooth = %s subtract(one_b, smooth)
  blend_left = %s multiply(inverse_smooth, scaled_inv_freq)
  blend_right = %s multiply(smooth, inv_freq_raw)
  blended_inv_freq = %s add(blend_left, blend_right)
  low_or_blend = %s select(low_band, scaled_inv_freq, blended_inv_freq)
  inv_freq = %s select(high_band, inv_freq_raw, low_or_blend)`,
		elementType, baseFreq,
		halfLiteral,
		elementType,
		halfLiteral,
		halfLiteral,
		elementType, scalingFactor,
		halfLiteral,
		halfLiteral,
		elementType, float64(originalContext),
		halfLiteral,
		elementType, lowFreqFactor,
		halfLiteral,
		elementType, highFreqFactor,
		halfLiteral,
		elementType, float64(originalContext)/lowFreqFactor,
		halfLiteral,
		elementType, float64(originalContext)/highFreqFactor,
		halfLiteral,
		predicateLiteral,
		predicateLiteral,
		halfLiteral,
		halfLiteral,
		halfLiteral,
		halfLiteral,
		elementType,
		halfLiteral,
		halfLiteral,
		halfLiteral,
		halfLiteral,
		halfLiteral,
		halfLiteral,
		halfLiteral), nil
}

func renderRoPERotationBody(
	elementType string,
	inputLiteral string,
	broadcastLiteral string,
	seqLen, numHeads, headDim, halfDim int,
	mode int,
) (string, error) {
	switch mode {
	case int(device.RoPEModeInterleaved):
		return renderInterleavedRoPERotationBody(
			elementType, inputLiteral, broadcastLiteral, seqLen, numHeads, headDim, halfDim,
		), nil
	case int(device.RoPEModeHalf):
		return renderHalfRoPERotationBody(
			elementType, inputLiteral, broadcastLiteral, seqLen, numHeads, headDim, halfDim,
		), nil
	default:
		return "", fmt.Errorf("xla rope: unsupported mode %d", mode)
	}
}

func renderInterleavedRoPERotationBody(
	elementType string,
	inputLiteral string,
	broadcastLiteral string,
	seqLen, numHeads, headDim, halfDim int,
) string {
	pairLiteral := fmt.Sprintf("%s[%d,%d,%d,2]{3,2,1,0}", elementType, seqLen, numHeads, halfDim)
	colLiteral := fmt.Sprintf("%s[%d,%d,%d,1]{3,2,1,0}", elementType, seqLen, numHeads, halfDim)

	return fmt.Sprintf(`  pairs = %s reshape(input), dimensions={%d,%d,%d,2}
  even = %s reshape(%s slice(pairs, {0,0,0,0}, {%d,%d,%d,1}, {1,1,1,1})), dimensions={%d,%d,%d}
  odd = %s reshape(%s slice(pairs, {0,0,0,1}, {%d,%d,%d,2}, {1,1,1,1})), dimensions={%d,%d,%d}
  out_even = %s subtract(%s multiply(even, cos_b), %s multiply(odd, sin_b))
  out_odd = %s add(%s multiply(even, sin_b), %s multiply(odd, cos_b))
  even_col = %s reshape(out_even), dimensions={%d,%d,%d,1}
  odd_col = %s reshape(out_odd), dimensions={%d,%d,%d,1}
  merged = %s concatenate({even_col, odd_col}), dimensions={3}
  ROOT result = %s reshape(merged), dimensions={%d,%d,%d}`,
		pairLiteral, seqLen, numHeads, halfDim,
		broadcastLiteral, pairLiteral, seqLen, numHeads, halfDim, seqLen, numHeads, halfDim,
		broadcastLiteral, pairLiteral, seqLen, numHeads, halfDim, seqLen, numHeads, halfDim,
		broadcastLiteral, broadcastLiteral, broadcastLiteral,
		broadcastLiteral, broadcastLiteral, broadcastLiteral,
		colLiteral, seqLen, numHeads, halfDim,
		colLiteral, seqLen, numHeads, halfDim,
		pairLiteral,
		inputLiteral, seqLen, numHeads, headDim)
}

func renderHalfRoPERotationBody(
	elementType string,
	inputLiteral string,
	broadcastLiteral string,
	seqLen, numHeads, headDim, halfDim int,
) string {
	return fmt.Sprintf(`  even = %s slice(input, {0,0,0}, {%d,%d,%d}, {1,1,1})
  odd = %s slice(input, {0,0,%d}, {%d,%d,%d}, {1,1,1})
  out_even = %s subtract(%s multiply(even, cos_b), %s multiply(odd, sin_b))
  out_odd = %s add(%s multiply(even, sin_b), %s multiply(odd, cos_b))
  ROOT result = %s concatenate({out_even, out_odd}), dimensions={2}`,
		broadcastLiteral, seqLen, numHeads, halfDim,
		broadcastLiteral, halfDim, seqLen, numHeads, headDim,
		broadcastLiteral, broadcastLiteral, broadcastLiteral,
		broadcastLiteral, broadcastLiteral, broadcastLiteral,
		inputLiteral)
}
