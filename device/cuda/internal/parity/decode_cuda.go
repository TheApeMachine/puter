//go:build cuda

package parity

import (
	"fmt"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/dtype/convert"
)

/*
DecodeFloat32Vector decodes encoded storage bytes to float32 lanes.
*/
func DecodeFloat32Vector(bytesIn []byte, format dtype.DType) []float32 {
	decoded, err := decodeVector(bytesIn, format)

	if err != nil {
		panic(err)
	}

	return decoded
}

func decodeVector(bytesIn []byte, format dtype.DType) ([]float32, error) {
	switch format {
	case dtype.Float32:
		if len(bytesIn)%4 != 0 {
			return nil, fmt.Errorf("cuda parity: invalid float32 byte length %d", len(bytesIn))
		}

		values := make([]float32, len(bytesIn)/4)

		for index := range values {
			values[index] = *(*float32)(unsafe.Pointer(&bytesIn[index*4]))
		}

		return values, nil
	case dtype.Float16:
		return convert.BytesToFloat32(format, bytesIn)
	case dtype.BFloat16:
		return convert.BytesToFloat32(format, bytesIn)
	default:
		return nil, fmt.Errorf("cuda parity: unsupported dtype %v", format)
	}
}
