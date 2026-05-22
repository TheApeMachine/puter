//go:build !darwin || !cgo

package runner

import (
	"fmt"

	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
)

func dispatchMultiAxisRoPE(node *ir.Node, args []tensor.Tensor) error {
	_ = node
	_ = args

	return fmt.Errorf("runner: multi_axis_rope dispatch requires darwin with cgo")
}
