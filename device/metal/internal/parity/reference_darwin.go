//go:build darwin && cgo

package parity

import (
	"fmt"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	cpuactivation "github.com/theapemachine/puter/device/cpu/activation"
)

const normEpsilon = 1e-5

/*
UnaryReference computes the CPU production reference for a unary activation.
*/
type UnaryReference func(dst, src unsafe.Pointer, count int)

/*
ReferenceReLU returns the production CPU reference kernel for ReLU.
*/
func ReferenceReLU(format dtype.DType) UnaryReference {
	return productionUnaryReference(cpuactivation.ReLU, format)
}

/*
ReferenceExp returns the production CPU reference kernel for Exp.
*/
func ReferenceExp(format dtype.DType) UnaryReference {
	return productionUnaryReference(cpuactivation.Exp, format)
}

/*
ReferenceGeluTanh returns the production CPU reference kernel for GeluTanh.
*/
func ReferenceGeluTanh(format dtype.DType) UnaryReference {
	return productionUnaryReference(cpuactivation.GeluTanh, format)
}

/*
ReferenceGelu returns the production CPU reference kernel for Gelu.
*/
func ReferenceGelu(format dtype.DType) UnaryReference {
	return productionUnaryReference(cpuactivation.Gelu, format)
}

func productionUnaryReference(
	kernel func(dst, src unsafe.Pointer, count int, format dtype.DType),
	format dtype.DType,
) UnaryReference {
	return func(dst, src unsafe.Pointer, count int) {
		kernel(dst, src, count, format)
	}
}

/*
ComputeUnaryReference runs the CPU reference into float32 lanes.
*/
func ComputeUnaryReference(
	source []float32,
	format dtype.DType,
	reference UnaryReference,
) []float32 {
	sourceBytes, err := encodeVector(source, format)

	if err != nil {
		panic(err)
	}

	destinationBytes := make([]byte, len(sourceBytes))
	reference(
		unsafe.Pointer(&destinationBytes[0]),
		unsafe.Pointer(&sourceBytes[0]),
		len(source),
	)

	decoded, err := decodeVector(destinationBytes, format)

	if err != nil {
		panic(err)
	}

	return decoded
}

/*
GroupNormReference computes the GroupNorm reference matching Metal MSL reduction order.
*/
func GroupNormReference(
	input, scale, bias []float32,
	batch, channels, spatial, groups int,
	storageDType dtype.DType,
) []float32 {
	output := make([]float32, len(input))
	channelsPerGroup := channels / groups
	groupSize := channelsPerGroup * spatial

	for batchIndex := 0; batchIndex < batch; batchIndex++ {
		for groupIndex := 0; groupIndex < groups; groupIndex++ {
			channelStart := groupIndex * channelsPerGroup
			groupStart := batchIndex*channels*spatial + channelStart*spatial
			groupInput := input[groupStart : groupStart+groupSize]
			groupOutput := output[groupStart : groupStart+groupSize]
			groupScale := scale[channelStart : channelStart+channelsPerGroup]
			groupBias := bias[channelStart : channelStart+channelsPerGroup]

			metalGroupNormGroup(groupInput, groupOutput, groupScale, groupBias, channelsPerGroup, spatial, storageDType)
		}
	}

	return output
}

/*
LayerNormReference computes the LayerNorm reference matching Metal MSL reduction order.
*/
func LayerNormReference(
	input, scale, bias []float32,
	rows, cols int,
	storageDType dtype.DType,
) []float32 {
	output := make([]float32, len(input))

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		rowStart := rowIndex * cols
		rowInput := input[rowStart : rowStart+cols]
		rowOutput := output[rowStart : rowStart+cols]

		metalLayerNormRow(rowInput, rowOutput, scale, bias, storageDType)
	}

	return output
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

/*
ComputeUnaryReferenceBytes runs the CPU reference and returns encoded storage bytes.
*/
func ComputeUnaryReferenceBytes(
	source []float32,
	format dtype.DType,
	reference UnaryReference,
) []byte {
	sourceBytes, err := encodeVector(source, format)

	if err != nil {
		panic(err)
	}

	destinationBytes := make([]byte, len(sourceBytes))
	reference(
		unsafe.Pointer(&destinationBytes[0]),
		unsafe.Pointer(&sourceBytes[0]),
		len(source),
	)

	return destinationBytes
}
