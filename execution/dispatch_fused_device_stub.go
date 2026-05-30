//go:build !darwin || !cgo

package execution

import (
	"github.com/theapemachine/manifesto/ast"
	"github.com/theapemachine/manifesto/codegen"
)

func (dispatcher *dispatcher) tryRunFusedOnMetalDevice(
	runner codegen.ElementwiseRunner,
	node *ast.GraphNode,
	inputSlots []int,
	outputSlot int,
) (bool, error) {
	_ = dispatcher
	_ = runner
	_ = node
	_ = inputSlots
	_ = outputSlot

	return false, nil
}
