//go:build cuda

package parity

import (
	"fmt"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/dtype/convert"
)

func encodeVector(values []float32, format dtype.DType) ([]byte, error) {
	switch format {
	case dtype.Float32:
		return convert.Float32ToBytes(values), nil
	case dtype.Float16:
		encoded := make([]dtype.F16, len(values))

		for index, value := range values {
			encoded[index] = dtype.Fromfloat32(value)
		}

		return convert.Float16ToBytes(encoded), nil
	case dtype.BFloat16:
		encoded := make([]dtype.BF16, len(values))

		for index, value := range values {
			encoded[index] = dtype.NewBfloat16FromFloat32(value)
		}

		return convert.BFloat16ToBytes(encoded), nil
	default:
		return nil, fmt.Errorf("cuda parity: unsupported dtype %v", format)
	}
}
