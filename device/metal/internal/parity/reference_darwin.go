//go:build darwin && cgo

package parity

import (
	"fmt"
	"math"
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
GroupNormReference computes the scalar GroupNorm reference in float32 space.
*/
func GroupNormReference(
	input, scale, bias []float32,
	batch, channels, spatial, groups int,
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

			mean := groupMeanFloat64(groupInput)
			variance := groupVarianceFloat64(groupInput, mean)
			mean32 := float32(mean)
			invStdDev := float32(1.0 / math.Sqrt(variance+float64(normEpsilon)))

			for channelIndex := 0; channelIndex < channelsPerGroup; channelIndex++ {
				channelOffset := channelIndex * spatial

				for spatialIndex := 0; spatialIndex < spatial; spatialIndex++ {
					index := channelOffset + spatialIndex
					normalized := (groupInput[index] - mean32) * invStdDev
					groupOutput[index] = normalized*groupScale[channelIndex] + groupBias[channelIndex]
				}
			}
		}
	}

	return output
}

/*
LayerNormReference computes the scalar LayerNorm reference in float32 space.
*/
func LayerNormReference(
	input, scale, bias []float32,
	rows, cols int,
) []float32 {
	output := make([]float32, len(input))

	for rowIndex := 0; rowIndex < rows; rowIndex++ {
		rowStart := rowIndex * cols
		rowInput := input[rowStart : rowStart+cols]
		rowOutput := output[rowStart : rowStart+cols]

		mean := groupMeanFloat64(rowInput)
		variance := groupVarianceFloat64(rowInput, mean)
		mean32 := float32(mean)
		invStdDev := float32(1.0 / math.Sqrt(variance+float64(normEpsilon)))

		for colIndex := 0; colIndex < cols; colIndex++ {
			normalized := (rowInput[colIndex] - mean32) * invStdDev
			rowOutput[colIndex] = normalized*scale[colIndex] + bias[colIndex]
		}
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

func groupMeanFloat64(values []float32) float64 {
	var sum float64

	for _, value := range values {
		sum += float64(value)
	}

	return sum / float64(len(values))
}

func groupVarianceFloat64(values []float32, mean float64) float64 {
	var sum float64

	for _, value := range values {
		delta := float64(value) - mean
		sum += delta * delta
	}

	return sum / float64(len(values))
}

func groupMean(values []float32) float32 {
	return float32(groupMeanFloat64(values))
}

func groupVariance(values []float32, mean float32) float32 {
	return float32(groupVarianceFloat64(values, float64(mean)))
}
