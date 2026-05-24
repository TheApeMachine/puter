package hlo

import (
	"fmt"

	"github.com/theapemachine/manifesto/dtype"
)

func RenderQuantInt8(
	moduleName string,
	elementFormat dtype.DType,
	count int,
	scale float32,
	zeroPoint int8,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	inputLiteral := fmt.Sprintf("%s[%d]{0}", elementType, count)
	outputLiteral := fmt.Sprintf("s8[%d]{0}", count)
	entryLayout := fmt.Sprintf("%s->%s", inputLiteral, outputLiteral)
	zeroPointFloat := float32(zeroPoint)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
  src = %s parameter(0)
  scale_c = %s[] constant(%g)
  scale_b = %s broadcast(scale_c), dimensions={0}
  scaled = %s divide(src, scale_b)
  zp_c = %s[] constant(%g)
  zp_b = %s broadcast(zp_c), dimensions={0}
  shifted = %s add(scaled, zp_b)
  rounded = %s round(shifted)
  min_c = %s[] constant(-128)
  min_b = %s broadcast(min_c), dimensions={0}
  max_c = %s[] constant(127)
  max_b = %s broadcast(max_c), dimensions={0}
  clamped_hi = %s minimum(rounded, max_b)
  clamped = %s maximum(clamped_hi, min_b)
  ROOT result = s8[%d]{0} convert(clamped)
}
`, moduleName, entryLayout,
		inputLiteral,
		elementType, scale, inputLiteral,
		inputLiteral, elementType, zeroPointFloat, inputLiteral,
		inputLiteral, elementType, inputLiteral,
		elementType, inputLiteral, elementType, inputLiteral,
		inputLiteral, elementType, inputLiteral,
		count), nil
}

func RenderDequantInt8(
	moduleName string,
	elementFormat dtype.DType,
	count int,
	scale float32,
	zeroPoint int8,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	inputLiteral := fmt.Sprintf("s8[%d]{0}", count)
	outputLiteral := fmt.Sprintf("%s[%d]{0}", elementType, count)
	entryLayout := fmt.Sprintf("%s->%s", inputLiteral, outputLiteral)
	zeroPointFloat := float32(zeroPoint)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
  src = %s parameter(0)
  converted = %s convert(src)
  zp_c = %s[] constant(%g)
  zp_b = %s broadcast(zp_c), dimensions={0}
  centered = %s subtract(converted, zp_b)
  scale_c = %s[] constant(%g)
  scale_b = %s broadcast(scale_c), dimensions={0}
  ROOT result = %s multiply(centered, scale_b)
}
`, moduleName, entryLayout,
		inputLiteral, outputLiteral,
		elementType, zeroPointFloat, outputLiteral,
		outputLiteral, elementType, scale, outputLiteral, outputLiteral), nil
}

func RenderDequantInt4(
	moduleName string,
	elementFormat dtype.DType,
	elementCount int,
	scale float32,
	zeroPoint int8,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	packedCount := (elementCount + 1) / 2
	inputLiteral := fmt.Sprintf("u8[%d]{0}", packedCount)
	outputLiteral := fmt.Sprintf("%s[%d]{0}", elementType, elementCount)
	entryLayout := fmt.Sprintf("%s->%s", inputLiteral, outputLiteral)
	zeroPointFloat := float32(zeroPoint)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
  packed = %s parameter(0)
  indices = s32[%d]{0} convert(iota(s32[%d]{0}), dimensions={0})
  pair_idx = s32[%d]{0} divide(indices, s32[] constant(2))
  bytes = u8[%d]{0} gather(packed, pair_idx),
    offset_dims={1},
    collapsed_slice_dims={0},
    start_index_map={0},
    index_vector_dim=1,
    slice_sizes={1}
  lo = u8[%d]{0} and(bytes, u8[] constant(15))
  hi = u8[%d]{0} shift-right-logical(bytes, u8[] constant(4))
  odd = pred[%d]{0} compare(and(indices, s32[] constant(1)), s32[] constant(0)), direction=NE
  nibble_u8 = u8[%d]{0} select(odd, hi, lo)
  nibble_s32 = s32[%d]{0} convert(nibble_u8)
  sign_mask = pred[%d]{0} compare(nibble_u8, u8[] constant(8)), direction=GE
  sign_adj = s32[%d]{0} multiply(convert(sign_mask), s32[] constant(16))
  signed = s32[%d]{0} subtract(nibble_s32, sign_adj)
  converted = %s convert(signed)
  zp_c = %s[] constant(%g)
  zp_b = %s broadcast(zp_c), dimensions={0}
  centered = %s subtract(converted, zp_b)
  scale_c = %s[] constant(%g)
  scale_b = %s broadcast(scale_c), dimensions={0}
  ROOT result = %s multiply(centered, scale_b)
}
`, moduleName, entryLayout,
		inputLiteral,
		elementCount, elementCount, elementCount, elementCount,
		elementCount, elementCount, elementCount, elementCount,
		elementCount, elementCount, elementCount, elementCount,
		elementType, elementType, zeroPointFloat, outputLiteral,
		outputLiteral, elementType, scale, outputLiteral, outputLiteral), nil
}
