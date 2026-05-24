package hlo

import (
	"fmt"

	"github.com/theapemachine/manifesto/dtype"
)

func RenderUpdateRepresentation(
	moduleName string,
	elementFormat dtype.DType,
	outDim, inDim int,
	learningRate float32,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	weightLiteral := fmt.Sprintf("%s[%d,%d]{1,0}", elementType, outDim, inDim)
	repLiteral := fmt.Sprintf("%s[%d]{0}", elementType, inDim)
	errorLiteral := fmt.Sprintf("%s[%d]{0}", elementType, outDim)
	outputLiteral := fmt.Sprintf("%s[%d]{0}", elementType, inDim)
	entryLayout := fmt.Sprintf("%s,%s,%s->%s", weightLiteral, repLiteral, errorLiteral, outputLiteral)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

%%add {
  lhs = %s[] parameter(0)
  rhs = %s[] parameter(1)
  ROOT result = %s[] add(lhs, rhs)
}

ENTRY main {
  weights = %s parameter(0)
  representation = %s parameter(1)
  prediction_error = %s parameter(2)
  error_row = %s[1,%d]{1,0} reshape(prediction_error), dimensions={1,%d}
  delta = %s dot(error_row, weights), contracting_dims={1}, lhs_batch_dims={}, rhs_batch_dims={}
  lr = %s[] constant(%g)
  lr_b = %s broadcast(lr), dimensions={0}
  scaled = %s multiply(delta, lr_b)
  ROOT result = %s add(representation, scaled)
}
`, moduleName, entryLayout,
		elementType, elementType, elementType,
		weightLiteral, repLiteral, errorLiteral,
		elementType, outDim, outDim, repLiteral,
		elementType, learningRate, repLiteral, repLiteral, repLiteral, repLiteral), nil
}

func RenderUpdateWeights(
	moduleName string,
	elementFormat dtype.DType,
	outDim, inDim int,
	learningRate float32,
) (string, error) {
	elementType, err := elementToken(elementFormat)

	if err != nil {
		return "", err
	}

	weightLiteral := fmt.Sprintf("%s[%d,%d]{1,0}", elementType, outDim, inDim)
	repLiteral := fmt.Sprintf("%s[%d]{0}", elementType, inDim)
	errorLiteral := fmt.Sprintf("%s[%d]{0}", elementType, outDim)
	entryLayout := fmt.Sprintf("%s,%s,%s->%s", weightLiteral, repLiteral, errorLiteral, weightLiteral)

	return fmt.Sprintf(`HloModule %s, entry_computation_layout={%s}

ENTRY main {
  weights = %s parameter(0)
  representation = %s parameter(1)
  prediction_error = %s parameter(2)
  error_col = %s[%d,1]{1,0} reshape(prediction_error), dimensions={%d,1}
  rep_row = %s[1,%d]{1,0} reshape(representation), dimensions={1,%d}
  outer = %s multiply(
    %s broadcast(error_col), dimensions={1},
    %s broadcast(rep_row), dimensions={0})
  lr = %s[] constant(%g)
  lr_b = %s broadcast(lr), dimensions={0,1}
  scaled = %s multiply(outer, lr_b)
  ROOT result = %s add(weights, scaled)
}
`, moduleName, entryLayout,
		weightLiteral, repLiteral, errorLiteral,
		elementType, outDim, outDim, elementType, inDim, inDim,
		weightLiteral, weightLiteral, weightLiteral, weightLiteral,
		elementType, learningRate, weightLiteral, weightLiteral, weightLiteral), nil
}
