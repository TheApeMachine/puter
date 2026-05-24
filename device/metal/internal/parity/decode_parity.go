package parity

import (
	"fmt"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
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
			return nil, fmt.Errorf("metal parity: invalid float32 byte length %d", len(bytesIn))
		}

		values := make([]float32, len(bytesIn)/4)

		for index := range values {
			values[index] = *(*float32)(unsafe.Pointer(&bytesIn[index*4]))
		}

		return values, nil
	case dtype.Float16:
		if len(bytesIn)%2 != 0 {
			return nil, fmt.Errorf("metal parity: invalid float16 byte length %d", len(bytesIn))
		}

		values := make([]float32, len(bytesIn)/2)

		for index := range values {
			value := dtype.F16(*(*uint16)(unsafe.Pointer(&bytesIn[index*2])))
			values[index] = value.Float32()
		}

		return values, nil
	case dtype.BFloat16:
		if len(bytesIn)%2 != 0 {
			return nil, fmt.Errorf("metal parity: invalid bfloat16 byte length %d", len(bytesIn))
		}

		values := make([]float32, len(bytesIn)/2)

		for index := range values {
			value := dtype.BF16(*(*uint16)(unsafe.Pointer(&bytesIn[index*2])))
			values[index] = value.Float32()
		}

		return values, nil
	default:
		return nil, fmt.Errorf("metal parity: unsupported dtype %v", format)
	}
}
