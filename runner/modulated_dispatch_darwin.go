//go:build darwin && cgo

package runner

import (
	"fmt"

	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device/metal"
)

func dispatchModulatedLayerNorm(node *ir.Node, args []tensor.Tensor) error {
	if len(args) != 3 {
		return fmt.Errorf("runner: modulated_layernorm requires input, modulation, and output tensors")
	}

	modulationSet, err := nodeIntAttribute(node, "set")
	if err != nil {
		return err
	}

	return metal.RunModulatedLayerNorm(args[0], args[1], args[2], modulationSet)
}

func dispatchGatedResidual(node *ir.Node, args []tensor.Tensor) error {
	if len(args) != 4 {
		return fmt.Errorf("runner: gated_residual requires residual, branch, modulation, and output tensors")
	}

	modulationSet, err := nodeIntAttribute(node, "set")
	if err != nil {
		return err
	}

	return metal.RunGatedResidual(args[0], args[1], args[2], args[3], modulationSet)
}
