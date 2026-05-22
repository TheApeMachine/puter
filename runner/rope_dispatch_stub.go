//go:build !darwin || !cgo

package runner

import (
	"fmt"

	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
)

func dispatchRoPE(node *ir.Node, args []tensor.Tensor) error {
	_ = node
	_ = args

	return fmt.Errorf("runner: rope dispatch requires darwin with cgo")
}
