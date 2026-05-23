package parity

import (
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
