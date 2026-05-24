package math

import (
	"fmt"
	"testing"
)

//go:noinline
func callFastExp32(x float32) float32 { return FastExp32(x) }

func TestCallFastExp32(t *testing.T) {
	x := float32(-18.256193)
	fmt.Printf("direct=%g call=%g\n", FastExp32(x), callFastExp32(x))
}
