package hlo

import (
	"fmt"

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
		indexCount, indexCount, indexCount, bagCount, bagCount, indexCount, bagCount,
		indexCount, indexCount, indexCount,
		elementType, outputLiteral, outputLiteral), nil
}
