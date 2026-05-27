package hlo

import (
	"fmt"
	"math"

	"github.com/theapemachine/manifesto/dtype"
)

func RenderEmbeddingLookup(
	moduleName string,
	elementFormat dtype.DType,
	vocab, hidden, indexCount int,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	tableLiteral := fmt.Sprintf("%s[%d,%d]{1,0}", elementType, vocab, hidden)
	indicesLiteral := fmt.Sprintf("s32[%d]{0}", indexCount)
	outputLiteral := fmt.Sprintf("%s[%d,%d]{1,0}", elementType, indexCount, hidden)
	entryLayout := fmt.Sprintf("%s,%s->%s", tableLiteral, indicesLiteral, outputLiteral)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
  table = %s parameter(0)
  indices = %s parameter(1)
  gather_idx = s32[%d,1]{1,0} reshape(indices), dimensions={0}
  ROOT result = %s gather(table, gather_idx),
    offset_dims={1},
    collapsed_slice_dims={0},
    start_index_map={0},
    index_vector_dim=1,
    slice_sizes={1,%d}
}
`, moduleName, entryLayout,
		tableLiteral, indicesLiteral,
		indexCount, outputLiteral, hidden), nil
}

func RenderEmbeddingBag(
	moduleName string,
	elementFormat dtype.DType,
	vocab, hidden, bagCount, indexCount int,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	tableLiteral := fmt.Sprintf("%s[%d,%d]{1,0}", elementType, vocab, hidden)
	indicesLiteral := fmt.Sprintf("s32[%d]{0}", indexCount)
	offsetsLiteral := fmt.Sprintf("s32[%d]{0}", bagCount)
	lookupLiteral := fmt.Sprintf("%s[%d,%d]{1,0}", elementType, indexCount, hidden)
	outputLiteral := fmt.Sprintf("%s[%d,%d]{1,0}", elementType, bagCount, hidden)
	entryLayout := fmt.Sprintf("%s,%s,%s->%s", tableLiteral, indicesLiteral, offsetsLiteral, outputLiteral)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

%%add {
  lhs = s32[] parameter(0)
  rhs = s32[] parameter(1)
  ROOT result = s32[] add(lhs, rhs)
}

ENTRY main {
  table = %s parameter(0)
  indices = %s parameter(1)
  offsets = %s parameter(2)
  gather_idx = s32[%d,1]{1,0} reshape(indices), dimensions={0}
  looked_up = %s gather(table, gather_idx),
    offset_dims={1},
    collapsed_slice_dims={0},
    start_index_map={0},
    index_vector_dim=1,
    slice_sizes={1,%d}
  pos = s32[%d]{0} convert(iota(s32[%d]{0}), dimensions={0})
  pos_col = s32[%d,1]{1,0} reshape(pos), dimensions={%d,1})
  offsets_row = s32[1,%d]{1,0} reshape(offsets), dimensions={1,%d})
  ge_mask = pred[%d,%d]{1,0} compare(pos_col, offsets_row), direction=GE
  bag_count = s32[%d]{0} reduce(ge_mask, s32[] constant(0)), dimensions={1}, to_apply=%%add
  bag_id = s32[%d]{0} subtract(bag_count, s32[] constant(1))
  scatter_idx = s32[%d,1]{1,0} reshape(bag_id), dimensions={0}
  init = %s[] constant(0)
  init_b = %s broadcast(init), dimensions={0,1}
  ROOT result = %s scatter(init_b, scatter_idx, looked_up),
    update_window_dims={1},
    inserted_window_dims={0},
    scatter_dims_to_operand_dims={0},
    index_vector_dim=1
}
`, moduleName, entryLayout,
		tableLiteral, indicesLiteral, offsetsLiteral,
		indexCount, lookupLiteral, hidden,
		indexCount, indexCount, indexCount, indexCount, bagCount, bagCount, indexCount, bagCount,
		indexCount, indexCount, indexCount,
		elementType, outputLiteral, outputLiteral), nil
}

func RenderTimestepEmbedding(
	moduleName string,
	elementFormat dtype.DType,
	count, dim int,
	maxPeriod float64,
	downscaleFreqShift float64,
	timestepDivisor float64,
	flipSinToCos bool,
) (string, error) {
	if elementFormat != dtype.Float32 {
		return "", fmt.Errorf("xla timestep embedding supports float32 output, got %s", elementFormat)
	}

	halfDim := dim / 2

	if halfDim <= 0 {
		return "", fmt.Errorf("xla timestep embedding dim must be at least 2, got %d", dim)
	}

	inputLiteral := fmt.Sprintf("f32[%d]{0}", count)
	halfLiteral := fmt.Sprintf("f32[%d,%d]{1,0}", count, halfDim)
	outputLiteral := fmt.Sprintf("f32[%d,%d]{1,0}", count, dim)
	entryLayout := fmt.Sprintf("%s->%s", inputLiteral, outputLiteral)
	scale := -math.Log(maxPeriod) / (float64(halfDim) - downscaleFreqShift)
	firstName := "sin_values"
	secondName := "cos_values"

	if flipSinToCos {
		firstName = "cos_values"
		secondName = "sin_values"
	}

	padding := ""
	root := fmt.Sprintf(
		"  ROOT result = %s concatenate(%s, %s), dimensions={1}",
		outputLiteral,
		firstName,
		secondName,
	)

	if dim%2 != 0 {
		padding = fmt.Sprintf(`
  zero = f32[] constant(0)
  zero_col = f32[%d,1]{1,0} broadcast(zero), dimensions={}
`, count)
		root = fmt.Sprintf(
			"  ROOT result = %s concatenate(%s, %s, zero_col), dimensions={1}",
			outputLiteral,
			firstName,
			secondName,
		)
	}

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
  timesteps = %s parameter(0)
  timestep_matrix = f32[%d,1]{1,0} reshape(timesteps)
  timestep_broadcast = %s broadcast(timestep_matrix), dimensions={0}
  divisor = f32[] constant(%.9g)
  divisor_broadcast = %s broadcast(divisor), dimensions={}
  scaled_timesteps = %s divide(timestep_broadcast, divisor_broadcast)
  freq_index = s32[%d]{0} iota(), iota_dimension=0
  freq_index_f32 = f32[%d]{0} convert(freq_index)
  scale = f32[] constant(%.9g)
  scale_broadcast = f32[%d]{0} broadcast(scale), dimensions={}
  exponent = f32[%d]{0} multiply(freq_index_f32, scale_broadcast)
  frequencies = f32[%d]{0} exponential(exponent)
  frequency_matrix = f32[1,%d]{1,0} reshape(frequencies)
  frequency_broadcast = %s broadcast(frequency_matrix), dimensions={1}
  angles = %s multiply(scaled_timesteps, frequency_broadcast)
  sin_values = %s sine(angles)
  cos_values = %s cosine(angles)%s
%s
}
`, moduleName, entryLayout,
		inputLiteral,
		count,
		halfLiteral,
		timestepDivisor,
		halfLiteral,
		halfLiteral,
		halfDim,
		halfDim,
		scale,
		halfDim,
		halfDim,
		halfDim,
		halfDim,
		halfLiteral,
		halfLiteral,
		halfLiteral,
		halfLiteral,
		padding,
		root), nil
}
