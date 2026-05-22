//go:build darwin && cgo

package runner

import (
	"fmt"

	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device/metal"
)

func dispatchRMSNorm(node *ir.Node, args []tensor.Tensor) error {
	if len(args) < 3 {
		return fmt.Errorf("runner: rmsnorm requires input, scale, and output tensors")
	}

	config := metal.RMSNormConfig{Epsilon: rmsNormEpsilonFromNode(node)}

	return metal.RunRMSNorm(args[0], args[1], args[len(args)-1], config)
}
