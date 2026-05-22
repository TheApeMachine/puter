//go:build !darwin || !cgo

package runner

import (
	"fmt"

	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
)

func dispatchModulatedLayerNorm(node *ir.Node, args []tensor.Tensor) error {
	_ = node
	_ = args

	return fmt.Errorf("runner: modulated_layernorm dispatch requires darwin with cgo")
}

func dispatchGatedResidual(node *ir.Node, args []tensor.Tensor) error {
	_ = node
	_ = args

	return fmt.Errorf("runner: gated_residual dispatch requires darwin with cgo")
}
