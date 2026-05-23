package hlo

import (
	"fmt"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func RenderDotProduct(
	moduleName string,
	elementFormat dtype.DType,
	leftShape tensor.Shape,
	rightShape tensor.Shape,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	leftLiteral := reductionInputLiteral(elementType, leftShape)
	rightLiteral := reductionInputLiteral(elementType, rightShape)
	entryLayout := fmt.Sprintf("%s,%s->%s[]", leftLiteral, rightLiteral, elementType)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

%%add {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT result = %s[] add(lhs, rhs)
}

ENTRY main {
  left = %s parameter(0)
  right = %s parameter(1)
  prod = %s multiply(left, right)
  init = %s[] constant(0)
  ROOT result = %s[] reduce(prod, init), dimensions={0}, to_apply=%%add
}
`, moduleName, entryLayout,
		elementType, elementType, elementType,
		leftLiteral, rightLiteral, leftLiteral, elementType, elementType), nil
}
