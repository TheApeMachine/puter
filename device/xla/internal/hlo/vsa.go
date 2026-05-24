package hlo

import (
	"fmt"

	"github.com/theapemachine/manifesto/dtype"
)

func RenderCyclicPermute(
	moduleName string,
	elementFormat dtype.DType,
	count int,
	shift int,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	if count == 0 {
		return "", fmt.Errorf("cyclic permute requires positive count")
	}

	normalizedShift := shift % count

	if normalizedShift < 0 {
		normalizedShift += count
	}

	vectorLiteral := fmt.Sprintf("%s[%d]{0}", elementType, count)
	entryLayout := fmt.Sprintf("%s->%s", vectorLiteral, vectorLiteral)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
  input = %s parameter(0)
  tail = %s slice(input), slice={[%d:%d]}
  head = %s slice(input), slice={[0:%d]}
  ROOT result = %s concatenate(tail, head), dimensions={0}
}
`, moduleName, entryLayout,
		vectorLiteral, vectorLiteral, normalizedShift, count,
		vectorLiteral, normalizedShift, vectorLiteral), nil
}
