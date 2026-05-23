//go:build xla

package parity

import (
	"github.com/theapemachine/manifesto/dtype"
	"github.com/theapemachine/manifesto/dtype/convert"
)

/*
DecodeFloat32Vector decodes encoded storage bytes to float32 lanes.
*/
func DecodeFloat32Vector(bytesIn []byte, format dtype.DType) []float32 {
	decoded, err := convert.BytesToFloat32(format, bytesIn)

	if err != nil {
		panic(err)
	}

	return decoded
}
