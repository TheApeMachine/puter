//go:build darwin && cgo

package runner

import (
	"fmt"

	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
	"github.com/theapemachine/puter/device/metal"
)

func dispatchMultiAxisRoPE(node *ir.Node, args []tensor.Tensor) error {
	if len(args) != 2 {
		return fmt.Errorf("runner: multi_axis_rope requires input and output tensors")
	}

	config := metal.MultiAxisRoPEConfig{}

	if latentSeqLen, ok := nodeOptionalIntAttribute(node, "latent_seq_len"); ok {
		config.LatentSeqLen = latentSeqLen
	}

	if latentSide, ok := nodeOptionalIntAttribute(node, "latent_side"); ok {
		config.LatentSide = latentSide
	}

	if base, ok := nodeOptionalFloatAttribute(node, "base"); ok {
		config.Base = float32(base)
	}

	return metal.RunMultiAxisRoPE(args[0], args[1], config)
}
