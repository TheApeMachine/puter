package layernorm

import (
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/puter/device/cpu/dot"
	"github.com/theapemachine/puter/device/cpu/elementwise"
	"github.com/theapemachine/puter/device/cpu/reduction"
)

func SumFloat32Native(values []float32) float32 {
	return reduction.SumFloat32Native(values)
}

func SumBFloat16Native(values []dtype.BF16) dtype.BF16 {
	return reduction.SumBFloat16Native(values)
}

func SumFloat16Native(values []dtype.F16) dtype.F16 {
	return reduction.SumFloat16Native(values)
}

func DotFloat32Native(left, right []float32) float32 {
	return dot.DotFloat32Native(left, right)
}

func DotBFloat16Native(left, right []dtype.BF16) dtype.BF16 {
	return dot.DotBFloat16Native(left, right)
}

func DotFloat16Native(left, right []dtype.F16) dtype.F16 {
	return dot.DotFloat16Native(left, right)
}

func MulFloat32Native(dst, left, right []float32) {
	elementwise.MulFloat32Native(dst, left, right)
}

func MulBFloat16Native(dst, left, right []dtype.BF16) {
	elementwise.MulBFloat16Native(dst, left, right)
}

func MulFloat16Native(dst, left, right []dtype.F16) {
	elementwise.MulFloat16Native(dst, left, right)
}
