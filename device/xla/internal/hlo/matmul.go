package hlo

import (
	"fmt"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func RenderMatmul(
	moduleName string,
	elementFormat dtype.DType,
	leftShape tensor.Shape,
	rightShape tensor.Shape,
	outputShape tensor.Shape,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	leftLiteral := reductionInputLiteral(elementType, leftShape)
	rightLiteral := reductionInputLiteral(elementType, rightShape)
	outputLiteral := reductionInputLiteral(elementType, outputShape)
	entryLayout := fmt.Sprintf("%s,%s->%s", leftLiteral, rightLiteral, outputLiteral)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
  lhs = %s parameter(0)
  rhs = %s parameter(1)
  ROOT result = %s dot(lhs, rhs),
    lhs_contracting_dimensions={1},
    rhs_contracting_dimensions={0}
}
`, moduleName, entryLayout, leftLiteral, rightLiteral, outputLiteral), nil
}
