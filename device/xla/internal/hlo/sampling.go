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
