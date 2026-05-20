package normalization

import "github.com/theapemachine/puter/device/cpu/reduction"

func SumFloat32Native(values []float32) float32 {
	return reduction.SumFloat32Native(values)
}
