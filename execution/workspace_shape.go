package execution

import (
	"fmt"
	"maps"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
)

func resolveShape(
	schema ir.ShapeSchema,
	dataType dtype.DType,
	bindings ir.SymbolMap,
) (tensor.Shape, error) {
	dimensions := make([]int, 0, len(schema.Dimensions))

	for index, dimension := range schema.Dimensions {
		if !dimension.IsSymbolic() {
			dimensions = append(dimensions, int(dimension.Static))
			continue
		}

		value, ok := bindings[dimension.Symbol]

		if !ok {
			return tensor.Shape{}, fmt.Errorf(
				"dim[%d] symbol %q unresolved at workspace materialization",
				index, dimension.Symbol,
			)
		}

		dimensions = append(dimensions, int(value))
	}

	_ = dataType

	return tensor.NewShape(dimensions)
}

func resolveLiveShape(
	schema ir.ShapeSchema,
	dataType dtype.DType,
	maxBindings ir.SymbolMap,
	launchBindings ir.SymbolMap,
) (tensor.Shape, error) {
	bindings := make(ir.SymbolMap, len(maxBindings)+len(launchBindings))
	maps.Copy(bindings, maxBindings)
	maps.Copy(bindings, launchBindings)

	return resolveShape(schema, dataType, bindings)
}
