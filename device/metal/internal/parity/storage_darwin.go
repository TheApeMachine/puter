//go:build darwin && cgo

package parity

import (
	"math"

	"github.com/theapemachine/manifesto/dtype"
)

func metalStorageLoad(value float32, format dtype.DType) float32 {
	switch format {
	case dtype.Float32:
		return value
	case dtype.Float16:
		return dtype.Fromfloat32(value).Float32()
	case dtype.BFloat16:
		bf16Value := dtype.NewBfloat16FromFloat32(value)
		return bf16Value.Float32()
	default:
		return value
	}
}

func metalStorageStore(value float32, format dtype.DType) float32 {
	return metalStorageLoad(value, format)
}

func metalFloat32InvStdDev(varianceSum float32, elementCount int) float32 {
	if elementCount == 0 {
		return 0
	}

	varianceMean := varianceSum / float32(elementCount)
	denominator := varianceMean + float32(normEpsilon)

	return float32(1.0) / float32(math.Sqrt(float64(denominator)))
}
