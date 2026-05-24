//go:build cuda

package parity

import (
	"math"
	"unsafe"

	"github.com/theapemachine/manifesto/dtype"
	cpuactivation "github.com/theapemachine/puter/device/cpu/activation"
	cpulayernorm "github.com/theapemachine/puter/device/cpu/layernorm"
	cpureduction "github.com/theapemachine/puter/device/cpu/reduction"
)

const normEpsilon = 1e-5

/*
UnaryReference computes the CPU production reference for a unary activation.
*/
type UnaryReference func(dst, src unsafe.Pointer, count int)

/*
DualParamReference computes the CPU production reference for dual-param activations.
*/
type DualParamReference func(dst, src unsafe.Pointer, count int, format dtype.DType, param0, param1 float32)

/*
SlopeParamReference computes the CPU production reference for slope-param activations.
*/
type SlopeParamReference func(dst, src unsafe.Pointer, count int, format dtype.DType, param float32)

/*
ReferenceReLU returns the production CPU reference kernel for ReLU.
*/
func ReferenceReLU(format dtype.DType) UnaryReference {
	return productionUnaryReference(cpuactivation.New().ReLU, format)
}

/*
ReferenceSnake returns the production CPU reference kernel for Snake.
*/
func ReferenceSnake(format dtype.DType, alpha float32) SlopeParamReference {
	return func(dst, src unsafe.Pointer, count int, storageDType dtype.DType, _ float32) {
		cpuactivation.New().Snake(dst, src, count, storageDType, alpha)
	}
}

/*
ReferenceHardTanhRange returns the production CPU reference kernel for HardTanhRange.
*/
func ReferenceHardTanhRange(format dtype.DType, minVal, maxVal float32) DualParamReference {
	return func(dst, src unsafe.Pointer, count int, storageDType dtype.DType, _, _ float32) {
		cpuactivation.New().HardTanhRange(dst, src, count, storageDType, minVal, maxVal)
	}
}

/*
ReferenceRReLU returns the production CPU reference kernel for RReLU.
*/
func ReferenceRReLU(format dtype.DType, lower, upper float32) DualParamReference {
	return func(dst, src unsafe.Pointer, count int, storageDType dtype.DType, _, _ float32) {
		cpuactivation.New().RReLU(dst, src, count, storageDType, lower, upper)
	}
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

/*
ComputeSlopeParamReferenceBytes runs a slope-param CPU reference into encoded bytes.
*/
func ComputeSlopeParamReferenceBytes(
	source []float32,
	format dtype.DType,
	reference SlopeParamReference,
	param float32,
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
		format,
		param,
	)

	return destinationBytes
}

/*
ComputeDualParamReferenceBytes runs a dual-param CPU reference into encoded bytes.
*/
func ComputeDualParamReferenceBytes(
	source []float32,
	format dtype.DType,
	reference DualParamReference,
	param0, param1 float32,
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
		format,
		param0,
		param1,
	)

	return destinationBytes
}

/*
LayerNormReference computes LayerNorm using the CPU generic row kernels.
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
		mean := cpureduction.SumFloat32Native(rowInput) / float32(cols)
		varianceSum := cpulayernorm.LayerNormSquaredDiffSumGeneric(rowInput, mean)
		invStdDev := float32(1.0) / float32(math.Sqrt(float64(varianceSum/float32(cols)+normEpsilon)))
		cpulayernorm.LayerNormApplyRowGeneric(rowOutput, rowInput, scale, bias, mean, invStdDev)
		_ = storageDType
	}

	return output
}
