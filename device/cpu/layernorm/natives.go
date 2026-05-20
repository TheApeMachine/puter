package layernorm

import (
	"github.com/theapemachine/puter/device/cpu/dot"
	"github.com/theapemachine/puter/device/cpu/elementwise"
	"github.com/theapemachine/puter/device/cpu/reduction"
)

func SumFloat32Native(values []float32) float32 {
	return reduction.SumFloat32Native(values)
}

func DotFloat32Native(left, right []float32) float32 {
	return dot.DotFloat32Native(left, right)
}

func MulFloat32Native(dst, left, right []float32) {
	elementwise.MulFloat32Native(dst, left, right)
}
