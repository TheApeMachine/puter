package hlo

import (
	"fmt"
	"strings"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/tensor"
)

func RenderReduction(
	moduleName string,
	elementFormat dtype.DType,
	inputShape tensor.Shape,
	reductionKind string,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	inputLiteral := reductionInputLiteral(elementType, inputShape)
	entryLayout := fmt.Sprintf("%s->%s[]", inputLiteral, elementType)

	switch reductionKind {
	case "sum":
		return renderReduceModule(moduleName, entryLayout, elementType, inputLiteral, "0", "add", "%add"), nil
	case "prod":
		return renderReduceModule(moduleName, entryLayout, elementType, inputLiteral, "1", "multiply", "%mul"), nil
	case "min":
		return renderReduceModule(moduleName, entryLayout, elementType, inputLiteral, "inf", "minimum", "%min"), nil
	case "max":
		return renderReduceModule(moduleName, entryLayout, elementType, inputLiteral, "-inf", "maximum", "%max"), nil
	case "l1norm":
		return renderL1NormModule(moduleName, entryLayout, elementType, inputLiteral), nil
	default:
		return "", fmt.Errorf("unsupported XLA reduction: %s", reductionKind)
	}
}

func renderReduceModule(
	moduleName string,
	entryLayout string,
	elementType string,
	inputLiteral string,
	initialValue string,
	combineOp string,
	computationName string,
) string {
	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

%s {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT result = %s[] %s(lhs, rhs)
}

ENTRY main {
  p0 = %s parameter(0)
  init = %s[] constant(%s)
  ROOT result = %s[] reduce(p0, init), dimensions={0}, to_apply=%s
}
`, moduleName, entryLayout,
		computationName, elementType, elementType, elementType, combineOp,
		inputLiteral, elementType, initialValue, elementType, computationName)
}

func renderL1NormModule(
	moduleName string,
	entryLayout string,
	elementType string,
	inputLiteral string,
) string {
	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

%%add {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT result = %s[] add(lhs, rhs)
}

ENTRY main {
  p0 = %s parameter(0)
  abs_val = %s abs(p0)
  init = %s[] constant(0)
  ROOT result = %s[] reduce(abs_val, init), dimensions={0}, to_apply=%%add
}
`, moduleName, entryLayout,
		elementType, elementType, elementType,
		inputLiteral, inputLiteral, elementType, elementType)
}

func reductionInputLiteral(elementType string, inputShape tensor.Shape) string {
	dimensions := inputShape.Dims()

	if len(dimensions) == 0 {
		return fmt.Sprintf("%s[]", elementType)
	}

	dimensionText := make([]string, len(dimensions))

	for index, dimension := range dimensions {
		dimensionText[index] = fmt.Sprintf("%d", dimension)
	}

	layout := reductionMinorToMajorLayout(len(dimensions))
	return fmt.Sprintf("%s[%s]{%s}", elementType, strings.Join(dimensionText, ","), layout)
}

func reductionMinorToMajorLayout(rank int) string {
	indices := make([]string, rank)

	for index := range rank {
		indices[index] = fmt.Sprintf("%d", rank-1-index)
	}

	return strings.Join(indices, ",")
}

func ReductionOperationName(kernel string) string {
	return "reduce_" + kernel
}

func ReductionKindFromOperation(operationName string) (string, bool) {
	if !strings.HasPrefix(operationName, "reduce_") {
		return "", false
	}

	return strings.TrimPrefix(operationName, "reduce_"), true
}
