//go:build xla

package parity

import (
	"fmt"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/dtype/convert"
)

/*
EncodeVector packs float32 lanes into native storage bytes for parity tests.
*/
func EncodeVector(values []float32, format dtype.DType) ([]byte, error) {
	return encodeVector(values, format)
}

func encodeVector(values []float32, format dtype.DType) ([]byte, error) {
	switch format {
	case dtype.Float64:
		encoded := make([]float64, len(values))

		for index, value := range values {
			encoded[index] = float64(value)
		}

		return convert.Float64ToBytes(encoded), nil
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
	case dtype.Float8E4M3:
		encoded := make([]dtype.F8E4M3, len(values))

		for index, value := range values {
			encoded[index] = dtype.NewF8E4M3FromFloat32(value)
		}

		return convert.Float8E4M3ToBytes(encoded), nil
	case dtype.Float8E5M2:
		encoded := make([]dtype.F8E5M2, len(values))

		for index, value := range values {
			encoded[index] = dtype.NewF8E5M2FromFloat32(value)
		}

		return convert.Float8E5M2ToBytes(encoded), nil
	default:
		return nil, fmt.Errorf("xla parity: unsupported dtype %v", format)
	}
}

func decodeFloat32Vector(bytesIn []byte, format dtype.DType) []float32 {
	return DecodeFloat32Vector(bytesIn, format)
}
