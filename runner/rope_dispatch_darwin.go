//go:build darwin && cgo

package runner

import (
	"fmt"

	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device/metal"
)

func dispatchRoPE(node *ir.Node, args []tensor.Tensor) error {
	if len(args) < 2 {
		return fmt.Errorf("runner: rope requires input and output tensors")
	}

	config := ropeConfigFromNode(node)

	return metal.RunRoPE(args[0], args[len(args)-1], config)
}
