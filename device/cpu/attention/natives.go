package attention

import (
	"github.com/theapemachine/puter/device/cpu/dot"
	"github.com/theapemachine/puter/device/cpu/elementwise"
	"github.com/theapemachine/puter/device/cpu/matmul"
	"github.com/theapemachine/puter/device/cpu/reduction"
)

func DotFloat32Native(left, right []float32) float32 {
	return dot.DotFloat32Native(left, right)
}

func ReduceMaxFloat32Native(values []float32) float32 {
	return reduction.ReduceMaxFloat32Native(values)
}

func MatmulFloat32Native(out, left, right []float32, rows, inner, cols int) {
	matmul.MatmulFloat32Native(out, left, right, rows, inner, cols)
}

func MulFloat32Native(dst, left, right []float32) {
	elementwise.MulFloat32Native(dst, left, right)
}

func AddFloat32Native(dst, left, right []float32) {
	elementwise.AddFloat32Native(dst, left, right)
}
