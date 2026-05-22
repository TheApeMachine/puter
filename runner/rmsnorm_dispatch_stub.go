//go:build !darwin || !cgo

package runner

import (
	"fmt"

	"github.com/theapemachine/manifesto/ir"
	"github.com/theapemachine/manifesto/tensor"
)

func dispatchRMSNorm(node *ir.Node, args []tensor.Tensor) error {
	_ = node
	_ = args

	return fmt.Errorf("runner: rmsnorm dispatch requires darwin with cgo")
}
